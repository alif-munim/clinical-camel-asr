#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transcribe_turbo.py – batch ASR with Whisper-v3-turbo + optional Silero VAD

FIXED (v2): Resolves the Tensor vs. NumPy array error during VAD chunk processing.
- Converts audio chunks to NumPy arrays before passing them to the pipeline.
- Hardcodes the model to "openai/whisper-large-v3-turbo" as requested.
"""

# ── sanity-check core deps ────────────────────────────────────────────────
import importlib, sys
from packaging import version
def _require(pkg, min_ver):
    mod = importlib.import_module(pkg)
    if version.parse(mod.__version__) < version.parse(min_ver):
        raise ImportError(f"{pkg}>={min_ver} required, found {mod.__version__}")
_require("transformers", "4.40.0")
_require("numba", "0.59.0")          # avoid the NumPy mismatch
# -------------------------------------------------------------------------
import transformers
import os, argparse, time, warnings
from pathlib import Path
import torch, librosa, numpy as np
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

warnings.filterwarnings("ignore")

# ── one-time global pipeline (prevents repeated GPU loads) ───────────────

_PIPE = None
def get_whisper_pipeline(
    model_name="openai/whisper-large-v3-turbo", # Model is now fixed
    device="cuda",
    torch_dtype=torch.float16,
    use_flash_attention=False,
    chunk_length_s=0,
    batch_size=1,
):
    global _PIPE
    if _PIPE is not None:
        return _PIPE                                             # ← reuse
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
    """
    Load the Silero VAD model and its utility functions from PyTorch Hub.
    """
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)
    
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    return model, get_speech_timestamps, read_audio

def get_audio_length(audio_file):
    """
    Calculate the length of an audio file in seconds using librosa.
    """
    try:
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception as e:
        print(f"Error getting audio length for {audio_file}: {e}")
        return None

def transcribe_audio_with_whisper(audio_file, pipe_config, use_vad=False):
    """
    Transcribe audio using Whisper, with optional VAD preprocessing.
    """
    pipe = pipe_config["instance"]
    generate_kwargs = {"return_timestamps": True}

    try:
        if use_vad:
            print("VAD enabled: Pre-processing audio to find speech segments...")
            try:
                vad_model, get_speech_timestamps, read_audio = load_silero_vad()
                SAMPLING_RATE = 16000
                wav = read_audio(audio_file, sampling_rate=SAMPLING_RATE)
                
                speech_timestamps = get_speech_timestamps(
                    wav, vad_model, 
                    sampling_rate=SAMPLING_RATE,
                    return_seconds=True
                )
                
                if not speech_timestamps:
                    print("VAD found no speech segments. Skipping transcription.")
                    return ""

                print(f"VAD found {len(speech_timestamps)} speech segment(s). Transcribing each segment...")
                
                full_transcription = []
                for segment in speech_timestamps:
                    start_time = segment['start']
                    end_time = segment['end']
                    
                    # Slice the torch.Tensor to get the speech chunk
                    chunk_tensor = wav[int(start_time * SAMPLING_RATE):int(end_time * SAMPLING_RATE)]
                    
                    # *** THE FIX IS HERE ***
                    # Convert the torch.Tensor chunk to a numpy.ndarray, which the pipeline expects
                    chunk_numpy = chunk_tensor.numpy()
                    
                    # Pass the raw waveform (as a dict with a NumPy array) to the pipeline
                    result = pipe(
                        {"raw": chunk_numpy, "sampling_rate": SAMPLING_RATE},
                        generate_kwargs=generate_kwargs
                    )
                    full_transcription.append(result['text'].strip())
                
                return " ".join(full_transcription)

            except Exception as vad_error:
                print(f"Error during VAD processing: {vad_error}. Falling back to standard transcription.")
                result = pipe(
                    audio_file, 
                    chunk_length_s=pipe_config["chunk_length_s"],
                    batch_size=pipe_config["batch_size"],
                    generate_kwargs=generate_kwargs
                )
                return result["text"]
        else:
            print("VAD disabled: Transcribing entire file...")
            result = pipe(
                audio_file, 
                chunk_length_s=pipe_config["chunk_length_s"],
                batch_size=pipe_config["batch_size"],
                generate_kwargs=generate_kwargs
            )
            return result["text"]
            
    except Exception as e:
        print(f"Error transcribing audio file {audio_file}: {e}")
        return None

def find_audio_files(input_dir, file_pattern="**/*.wav"):
    """
    Find all audio files in the directory matching the pattern (recursive).
    """
    # Using rglob for recursive search, which is more intuitive
    return sorted(list(Path(input_dir).rglob(file_pattern)))

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files with Whisper and optional VAD.")
    parser.add_argument("--input_dir", required=True, help="Directory containing audio files (e.g., WAV, MP3, FLAC).")
    parser.add_argument("--output_dir", required=True, help="Directory to save transcription text files.")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2 for faster inference (requires compatible hardware).")
    parser.add_argument("--chunk_length", type=int, default=30, help="Chunk length in seconds for long-form transcription (used when VAD is disabled).")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for long-form transcription (used when VAD is disabled).")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for no limit).")
    parser.add_argument("--use_vad", action="store_true", help="Use Silero VAD to detect speech and transcribe only speech segments.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, with dtype: {torch_dtype}")
    print(f"Using Transformers version: {transformers.__version__}")
    
    audio_files = find_audio_files(args.input_dir)
    print(f"Found {len(audio_files)} audio files to process.")
    
    if args.use_vad:
        print("Voice Activity Detection (VAD) is ENABLED. Transcription will focus only on detected speech segments.")
    else:
        print("Voice Activity Detection (VAD) is DISABLED. The entire audio file will be transcribed.")

    if args.limit > 0:
        audio_files = audio_files[:args.limit]
        print(f"Limiting processing to the first {len(audio_files)} files.")
    
    # Model name is now hardcoded as per your request
    model_name = "openai/whisper-large-v3-turbo"
    
    if args.use_flash_attention and device == "cuda:0":
        print("Flash Attention 2 is ENABLED.")
    
    # Initialize the pipeline once
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
        
        transcription = transcribe_audio_with_whisper(
            str(audio_file),
            pipe_config,
            args.use_vad
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