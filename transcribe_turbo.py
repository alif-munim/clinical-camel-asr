#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transcribe_turbo.py – batch ASR with Whisper-v3-turbo + optional Silero VAD

MODIFIED (v8): Implements a prompting strategy to preserve context between VAD segments.
The transcription of each segment is used as a prompt for the next, improving
coherence and accuracy for VAD-enabled transcriptions.
"""

# ── sanity-check core deps ────────────────────────────────────────────────
import importlib, sys
from packaging import version
def _require(pkg, min_ver):
    mod = importlib.import_module(pkg)
    if version.parse(mod.__version__) < version.parse(min_ver):
        raise ImportError(f"{pkg}>={min_ver} required, found {mod.__version__}")
_require("transformers", "4.40.0")
_require("numba", "0.59.0")
# -------------------------------------------------------------------------
import transformers
import os, argparse, time, warnings
from pathlib import Path
from typing import Optional
import torch, librosa, numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
# **FIX:** Import WhisperNoSpeechDetection from its correct submodule
from transformers.generation.logits_process import WhisperNoSpeechDetection


warnings.filterwarnings("ignore")

# --- MONKEY-PATCH FOR TRANSFORMERS BUG ---
# This patch is still required to fix an internal bug in the transformers library
# when using the no_speech_threshold parameter, as the pipeline calls the same underlying code.
def _fixed_set_inputs(self, inputs):
    if "inputs" in inputs:
        inputs["input_features"] = inputs.pop("inputs")
    if "input_ids" in inputs:
        inputs["decoder_input_ids"] = inputs.pop("input_ids")
    self.inputs = inputs
WhisperNoSpeechDetection.set_inputs = _fixed_set_inputs
# -----------------------------------------


# ── one-time global pipeline (prevents repeated GPU loads) ───────────────
_PIPE = None
def get_whisper_pipeline(
    model_name="openai/whisper-large-v3-turbo",
    device="cuda",
    torch_dtype=torch.float16,
    use_flash_attention=False,
    chunk_length_s=30,
    batch_size=24,
):
    """Initializes and returns the Whisper pipeline, loading it only once."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    model_args = dict(
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if use_flash_attention:
        model_args["attn_implementation"] = "flash_attention_2"

    print(f"Initializing model: {model_name}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **model_args)
    model.to(device)

    proc = AutoProcessor.from_pretrained(model_name)
    pipe_args = dict(
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    pipe_instance = pipeline("automatic-speech-recognition", **pipe_args)

    _PIPE = {
        "instance": pipe_instance,
        "chunk_length_s": chunk_length_s,
        "batch_size": batch_size
    }
    return _PIPE
# ------------------------------------------------------------------------

### Helper functions
def load_silero_vad():
    """Loads the Silero VAD model and utilities from torch.hub."""
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)
    (get_speech_timestamps, _, read_audio, *_) = utils
    return model, get_speech_timestamps, read_audio

def get_audio_length(audio_file):
    """Returns the duration of an audio file in seconds."""
    try:
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception as e:
        print(f"Error getting audio length for {audio_file}: {e}")
        return None

# ==============================================================================
# Main Transcription Function
# ==============================================================================
def transcribe_audio_with_whisper(
    audio_file: str,
    pipe_config: dict,
    language: str = "english",
    use_vad: bool = False,
    compression_ratio_threshold: Optional[float] = None,
    logprob_threshold: Optional[float] = None,
    no_speech_threshold: Optional[float] = None,
    vad_threshold: float = 0.5,
    vad_min_speech_duration_ms: int = 250,
    vad_min_silence_duration_ms: int = 100,
    vad_speech_pad_ms: int = 30,
):
    """
    Transcribe audio using Whisper, with optional VAD and generation parameters.
    This version uses the robust pipeline method for long-form transcription.
    """
    pipe = pipe_config["instance"]
    
    # Build the dictionary for optional generation args for Whisper.
    generate_kwargs = {}
    
    # Add the language to the generation arguments. This is the correct way
    # to force a specific language and disable multilingual auto-detection.
    if language:
        generate_kwargs["language"] = language
    
    if compression_ratio_threshold is not None:
        generate_kwargs["compression_ratio_threshold"] = compression_ratio_threshold
    if logprob_threshold is not None:
        generate_kwargs["logprob_threshold"] = logprob_threshold
    if no_speech_threshold is not None:
        generate_kwargs["no_speech_threshold"] = no_speech_threshold
    
    # Add default temperature fallback and conditioning for robustness
    generate_kwargs["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    generate_kwargs["condition_on_prev_tokens"] = True
    
    try:
        if use_vad:
            # The VAD path processes audio by finding speech chunks first.
            print("VAD enabled: Pre-processing audio to find speech segments...")
            vad_model, get_speech_timestamps, read_audio = load_silero_vad()
            SAMPLING_RATE = 16000
            wav = read_audio(audio_file, sampling_rate=SAMPLING_RATE)
            
            # Use the provided VAD parameters
            speech_timestamps = get_speech_timestamps(
                wav,
                vad_model,
                threshold=vad_threshold,
                min_speech_duration_ms=vad_min_speech_duration_ms,
                min_silence_duration_ms=vad_min_silence_duration_ms,
                speech_pad_ms=vad_speech_pad_ms,
                sampling_rate=SAMPLING_RATE,
                return_seconds=True
            )
            if not speech_timestamps:
                print("VAD found no speech segments. Skipping transcription.")
                return ""
            
            print(f"VAD found {len(speech_timestamps)} speech segment(s). Transcribing with context preservation...")
            
            # --- START OF MODIFICATIONS FOR CONTEXT PRESERVATION ---
            
            # 1. Initialize a list to hold the text from each transcribed chunk.
            transcribed_chunks = []
            
            for i, segment in enumerate(speech_timestamps):
                chunk_tensor = wav[int(segment['start'] * SAMPLING_RATE):int(segment['end'] * SAMPLING_RATE)]
                # The pipeline can handle numpy arrays directly
                chunk_numpy = chunk_tensor.numpy()
                
                # 2. Create a prompt from all previously transcribed text.
                prompt = " ".join(transcribed_chunks)
                
                # 3. Create a copy of the generation args and add the prompt.
                #    This ensures the prompt from one chunk is passed to the next.
                current_generate_kwargs = generate_kwargs.copy()
                current_generate_kwargs["prompt"] = prompt
                
                print(f"  - Transcribing segment {i+1}/{len(speech_timestamps)}...")
                result = pipe(chunk_numpy, generate_kwargs=current_generate_kwargs)
                new_text = result['text'].strip()
                
                # 4. Append the newly transcribed text to our list for the next iteration.
                transcribed_chunks.append(new_text)

            # 5. Join all the transcribed chunks together at the very end.
            return " ".join(transcribed_chunks)
            
            # --- END OF MODIFICATIONS ---

        else:
            # This is the robust implementation for non-VAD, long-form audio
            print("VAD disabled: Transcribing entire file with chunking...")
            
            result = pipe(
                audio_file,
                chunk_length_s=pipe_config["chunk_length_s"],
                batch_size=pipe_config["batch_size"],
                return_timestamps=True,
                generate_kwargs=generate_kwargs
            )
            return result["text"]
            
    except Exception as e:
        print(f"Error transcribing audio file {audio_file}: {e}")
        return None
# ==============================================================================
# End of Transcription Function
# ==============================================================================

def find_audio_files(input_dir, file_pattern="**/*.wav"):
    """Finds all audio files in a directory matching a pattern."""
    return sorted(list(Path(input_dir).rglob(file_pattern)))

def main():
    """Main function to parse arguments and process audio files."""
    parser = argparse.ArgumentParser(description="Batch transcribe audio files with Whisper and optional VAD.")
    
    # --- I/O Arguments ---
    parser.add_argument("--input_dir", required=True, help="Directory containing audio files (e.g., WAV, MP3, FLAC).")
    parser.add_argument("--output_dir", required=True, help="Directory to save transcription text files.")
    
    # --- Performance Arguments ---
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2 for faster inference (requires compatible hardware).")
    parser.add_argument("--chunk_length", type=int, default=30, help="Chunk length in seconds for long-form transcription (used when VAD is disabled).")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for long-form transcription (used when VAD is disabled).")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for no limit).")
    
    # --- VAD Control Arguments ---
    parser.add_argument("--use_vad", action="store_true", help="Use Silero VAD to detect speech and transcribe only speech segments.")
    parser.add_argument("--vad_threshold", type=float, default=0.5, help="Speech probability threshold for VAD. Default: 0.5")
    parser.add_argument("--vad_min_speech_duration_ms", type=int, default=250, help="VAD: Minimum speech duration in ms. Default: 250")
    parser.add_argument("--vad_min_silence_duration_ms", type=int, default=100, help="VAD: Minimum silence duration in ms. Default: 100")
    parser.add_argument("--vad_speech_pad_ms", type=int, default=30, help="VAD: Padding on each side of speech segment in ms. Default: 30")

    # --- Whisper Generation Arguments ---
    parser.add_argument("--language", type=str, default="english", help="Language for transcription (e.g., 'en', 'es'). Default: 'english'.")
    parser.add_argument("--compression_ratio_threshold", type=float, default=2.4, help="Whisper: If gzip compression ratio is > this value, treat as background noise. Default: 2.4")
    parser.add_argument("--logprob_threshold", type=float, default=-1.0, help="Whisper: If avg log probability is < this value, treat as silence. Default: -1.0")
    parser.add_argument("--no_speech_threshold", type=float, default=0.6, help="Whisper: If <|nospeech|> token probability is > this value, suppress segment. Default: 0.6")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, with dtype: {torch_dtype}")
    print(f"Using Transformers version: {transformers.__version__}")
    
    audio_files = find_audio_files(args.input_dir)
    print(f"Found {len(audio_files)} audio files to process.")
    
    if args.use_vad:
        print("Voice Activity Detection (VAD) is ENABLED.")
        print(f"  - VAD Settings: threshold={args.vad_threshold}, min_speech={args.vad_min_speech_duration_ms}ms, min_silence={args.vad_min_silence_duration_ms}ms, padding={args.vad_speech_pad_ms}ms")
    else:
        print("Voice Activity Detection (VAD) is DISABLED. The entire audio file will be transcribed.")

    if args.limit > 0:
        audio_files = audio_files[:args.limit]
        print(f"Limiting processing to the first {len(audio_files)} files.")
    
    model_name = "openai/whisper-large-v3-turbo"
    
    if args.use_flash_attention and device == "cuda:0":
        print("Flash Attention 2 is ENABLED.")
    
    pipe_config = get_whisper_pipeline(
        model_name=model_name,
        device=device,
        torch_dtype=torch_dtype,
        use_flash_attention=args.use_flash_attention,
        chunk_length_s=args.chunk_length,
        batch_size=args.batch_size
    )
    
    total_time = 0
    total_audio_length = 0
    successful_files = 0
    
    for i, audio_file in enumerate(audio_files):
        stem = audio_file.stem
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {stem}")
        
        start_time = time.time()
        
        audio_length = get_audio_length(audio_file)
        if audio_length:
            print(f"Audio duration: {audio_length:.2f} seconds")
            total_audio_length += audio_length
        
        # Pass all the arguments to the transcription function
        transcription = transcribe_audio_with_whisper(
            str(audio_file),
            pipe_config,
            language=args.language,
            use_vad=args.use_vad,
            compression_ratio_threshold=args.compression_ratio_threshold,
            logprob_threshold=args.logprob_threshold,
            no_speech_threshold=args.no_speech_threshold,
            vad_threshold=args.vad_threshold,
            vad_min_speech_duration_ms=args.vad_min_speech_duration_ms,
            vad_min_silence_duration_ms=args.vad_min_silence_duration_ms,
            vad_speech_pad_ms=args.vad_speech_pad_ms
        )
        
        if transcription is None:
            print(f"Skipping {stem} due to a transcription error.")
            continue
        
        output_path = Path(args.output_dir) / f"{stem}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        processing_time = time.time() - start_time
        total_time += processing_time
        successful_files += 1
        
        print(f"Transcription saved to: {output_path}")
        print(f"Time taken: {processing_time:.2f} seconds")
        
        if audio_length and audio_length > 0:
            rtf = processing_time / audio_length
            print(f"Real-Time Factor (RTF): {rtf:.2f}")
    
    print("\n" + "="*50)
    print("Batch Processing Summary")
    print("="*50)
    if successful_files > 0:
        print(f"Successfully transcribed {successful_files} of {len(audio_files)} files.")
        print(f"Total processing time: {total_time:.2f} seconds.")
        if total_audio_length > 0:
            overall_rtf = total_time / total_audio_length
            print(f"Total audio processed: {total_audio_length:.2f} seconds.")
            print(f"Overall Real-Time Factor (RTF): {overall_rtf:.2f}")
    else:
        print("No files were successfully processed.")

if __name__ == "__main__":
    main()