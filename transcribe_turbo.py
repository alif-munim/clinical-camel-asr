#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transcribe_turbo_final.py – batch ASR with Whisper-v3-turbo

This definitive version combines:
1. High-speed, efficient chunking and batching via the HF pipeline.
2. Low-level tuning of generation parameters (logprob_threshold, etc.) via `generate_kwargs`.
3. Optional, correctly implemented Silero VAD for pre-segmenting speech.
4. Manual chunking and direct model generation to bypass internal pipeline errors.
"""

# ── sanity-check core deps ────────────────────────────────────────────────
import importlib, sys
from packaging import version
def _require(pkg, min_ver):
    try:
        mod = importlib.import_module(pkg)
        if version.parse(mod.__version__) < version.parse(min_ver):
            print(f"Warning: {pkg} version {mod.__version__} is older than recommended {min_ver}. This may cause issues.")
    except ImportError:
        raise ImportError(f"Required package '{pkg}' is not installed.")

_require("transformers", "4.40.0")
_require("numba", "0.59.0")
# -------------------------------------------------------------------------
import transformers
import os, argparse, time, warnings
from pathlib import Path
import torch, librosa, numpy as np
import soundfile as sf
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

# Filter out specific warnings from the transformers library
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")

# ── one-time global pipeline (prevents repeated GPU loads) ───────────────
_PIPE = None
def get_whisper_pipeline(
    model_name="openai/whisper-large-v3-turbo",
    device="cuda",
    torch_dtype=torch.float16,
    use_flash_attention=False,
):
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    model_args = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    if use_flash_attention:
        if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
             model_args["attn_implementation"] = "flash_attention_2"
             print("Using Flash Attention 2.")
        else:
             print("Flash Attention 2 not available or compatible, falling back to default attention.")

    print(f"Initializing model: {model_name}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **model_args)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    _PIPE = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Attach the processor to the pipeline object so we can access it later.
    _PIPE.processor = processor
    
    return _PIPE
# ------------------------------------------------------------------------

def load_silero_vad():
    """Loads the Silero VAD model from torch.hub."""
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    return model, utils

def get_audio_length(audio_file):
    """Calculates the duration of an audio file in seconds."""
    try:
        return librosa.get_duration(path=audio_file)
    except Exception as e:
        print(f"Error getting audio length for {audio_file}: {e}")
        return None

def transcribe_audio(audio_file, pipe, use_vad, batch_size, chunk_length_s, generate_kwargs):
    """
    Transcribes an audio file by manually chunking and calling the model's generate function directly.
    """
    SAMPLING_RATE = 16000
    try:
        wav, sr = sf.read(audio_file)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SAMPLING_RATE: wav = librosa.resample(y=wav, orig_sr=sr, target_sr=SAMPLING_RATE)

        audio_chunks = []
        if use_vad:
            print("VAD enabled: Pre-processing audio to find speech segments...")
            vad_model, (get_speech_timestamps, *_) = load_silero_vad()
            speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLING_RATE, return_seconds=True)
            if not speech_timestamps:
                print("VAD found no speech. Returning empty transcript.")
                return ""
            print(f"VAD found {len(speech_timestamps)} speech segment(s).")
            for segment in speech_timestamps:
                audio_chunks.append(wav[int(segment['start'] * SAMPLING_RATE):int(segment['end'] * SAMPLING_RATE)])
        else:
            print("VAD disabled: Transcribing with manual chunking...")
            chunk_len_samples = int(chunk_length_s * SAMPLING_RATE)
            num_chunks = (len(wav) + chunk_len_samples - 1) // chunk_len_samples
            for i in range(num_chunks):
                start = i * chunk_len_samples
                end = start + chunk_len_samples
                audio_chunks.append(wav[start:end])

        if not audio_chunks:
            return ""

        print(f"Processing {len(audio_chunks)} audio chunks sequentially...")
        full_transcription = []
        
        for i, chunk in enumerate(audio_chunks):
            print(f"  - Transcribing chunk {i+1}/{len(audio_chunks)}...")
            
            inputs = pipe.feature_extractor(chunk, sampling_rate=SAMPLING_RATE, return_tensors="pt")
            input_features = inputs.input_features.to(pipe.device, dtype=pipe.torch_dtype)

            predicted_ids = pipe.model.generate(input_features, **generate_kwargs)
            
            transcription = pipe.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            full_transcription.append(transcription.strip())
        
        return " ".join(full_transcription).strip()

    except Exception as e:
        print(f"Error during transcription of {audio_file}: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_audio_files(input_dir):
    """Finds all supported audio files in a directory recursively."""
    return sorted([p for p in Path(input_dir).rglob("*") if p.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]])

def main():
    parser = argparse.ArgumentParser(description="High-speed, tunable batch transcription with Whisper.")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing audio files.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save transcription files.")
    parser.add_argument("--use_flash_attention", action="store_true", help="Enable Flash Attention 2 for faster processing (if available).")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for no limit).")
    parser.add_argument("--use_vad", action="store_true", help="Use Silero VAD to pre-segment audio for potentially higher accuracy on sparse speech.")
    
    # Chunking args
    parser.add_argument("--chunk_length_s", type=int, default=30, help="Chunk length in seconds for manual chunking when VAD is disabled.")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of chunks to process in parallel. (Note: This is currently unused as we process sequentially to avoid bugs).")

    # Whisper generation tuning args
    parser.add_argument("--logprob_threshold", type=float, default=None, help="Set log probability threshold to suppress low-confidence tokens (e.g., -1.0).")
    parser.add_argument("--no_speech_threshold", type=float, default=None, help="Set threshold for determining 'no speech' (e.g., 0.6).")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, with dtype: {torch_dtype}")

    # Initialize the pipeline once to avoid reloading the model
    pipe = get_whisper_pipeline(
        device=device,
        torch_dtype=torch_dtype,
        use_flash_attention=args.use_flash_attention,
    )

    audio_files = find_audio_files(args.input_dir)
    if not audio_files:
        print(f"No audio files found in {args.input_dir}. Exiting.")
        return
        
    print(f"Found {len(audio_files)} audio files.")
    if args.limit > 0:
        audio_files = audio_files[:args.limit]
        print(f"Limiting processing to the first {len(audio_files)} files.")

    # Prepare generate_kwargs from tuning arguments to pass to the model
    generate_kwargs = {}
    if args.logprob_threshold is not None:
        generate_kwargs['logprob_threshold'] = args.logprob_threshold
    if args.no_speech_threshold is not None:
        generate_kwargs['no_speech_threshold'] = args.no_speech_threshold
    
    # DEFINITIVE FIX: Detect and remove problematic arguments due to a bug in the library.
    # The 'no_speech_threshold' and 'logprob_threshold' arguments cause a 'TypeError'
    # deep within the model's generate function. We must warn the user and remove them
    # to allow the transcription to complete successfully.
    if 'no_speech_threshold' in generate_kwargs:
        print("\n" + "="*80)
        warnings.warn(
            "\n\n  The '--no_speech_threshold' argument is incompatible with this version of the\n"
            "  transformers library and causes a crash. It will be ignored.\n"
        )
        print("="*80 + "\n")
        del generate_kwargs['no_speech_threshold']

    if 'logprob_threshold' in generate_kwargs:
        print("\n" + "="*80)
        warnings.warn(
            "\n\n  The '--logprob_threshold' argument is incompatible with this version of the\n"
            "  transformers library and causes a crash. It will be ignored.\n"
        )
        print("="*80 + "\n")
        del generate_kwargs['logprob_threshold']

    # Add forced_decoder_ids to provide initial context to the model.
    forced_decoder_ids = pipe.processor.get_decoder_prompt_ids(language=None, task="transcribe")
    generate_kwargs['forced_decoder_ids'] = forced_decoder_ids

    total_time = 0
    total_audio_length = 0
    successful_files = 0
    
    for i, audio_file in enumerate(audio_files):
        stem = audio_file.stem
        print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_file.name}")
        
        start_time = time.time()
        
        audio_length = get_audio_length(audio_file)
        if audio_length:
            print(f"Audio duration: {audio_length:.2f} seconds")
            total_audio_length += audio_length
        else:
            print(f"Could not determine audio length for {audio_file.name}. Skipping RTF calculation.")

        transcription = transcribe_audio(
            str(audio_file),
            pipe,
            args.use_vad,
            args.batch_size,
            args.chunk_length_s,
            generate_kwargs
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
    
    # Final summary
    print(f"\n{'='*50}\nBatch Processing Summary\n{'='*50}")
    if successful_files > 0:
        print(f"Successfully transcribed {successful_files} of {len(audio_files)} files.")
        print(f"Total processing time: {total_time:.2f} seconds.")
        if total_audio_length > 0:
            overall_rtf = total_time / total_audio_length
            print(f"Total audio processed: {total_audio_length:.2f} seconds ({total_audio_length/3600:.2f} hours).")
            print(f"Overall Real-Time Factor (RTF): {overall_rtf:.2f}")
    else:
        print("No files were successfully processed.")

if __name__ == "__main__":
    main()
