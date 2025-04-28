#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
transcribe_turbo.py – batch ASR with Whisper-v3-turbo + optional Silero VAD
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
    model_name="openai/whisper-large-v3-turbo",
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

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **model_args)
    model.to(device)

    proc = AutoProcessor.from_pretrained(model_name)
    pipe_args = dict(
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"return_timestamps": True},
    )
    if chunk_length_s > 0:
        pipe_args.update(chunk_length_s=chunk_length_s, batch_size=batch_size)

    _PIPE = pipeline("automatic-speech-recognition", **pipe_args)
    return _PIPE
# ------------------------------------------------------------------------

### Helper functions (everything below here is exactly your original code)

def load_silero_vad():
    """
    Load the Silero VAD model
    """
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False)
    
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    return model, get_speech_timestamps, read_audio

def get_audio_length(audio_file):
    """
    Calculate the length of an audio file in seconds
    """
    try:
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception as e:
        print(f"Error getting audio length for {audio_file}: {e}")
        return None

def transcribe_audio_with_whisper(audio_file, model_name="openai/whisper-large-v3-turbo", device="cuda", 
                                  torch_dtype=torch.float16, use_flash_attention=False, 
                                  chunk_length_s=0, batch_size=1, use_vad=False):
    """
    Transcribe audio using Whisper model, optionally with VAD preprocessing
    """
    try:
        # Additional args for model loading
        model_args = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }
        
        # Add flash attention if requested
        if use_flash_attention:
            model_args["attn_implementation"] = "flash_attention_2"
        
        # Load model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, 
            **model_args
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Setup pipeline parameters
        pipe_args = {
            "model": model,
            "tokenizer": processor.tokenizer,
            "feature_extractor": processor.feature_extractor,
            "torch_dtype": torch_dtype,
            "device": device,
        }
        
        # Add chunking if specified
        if chunk_length_s > 0:
            pipe_args["chunk_length_s"] = chunk_length_s
            pipe_args["batch_size"] = batch_size
        
        pipe = pipeline(
            "automatic-speech-recognition",
            **pipe_args,
            generate_kwargs={"return_timestamps": True}
        )

        # If using VAD, process the audio segments
        if use_vad:
            try:
                print("Using VAD to detect speech segments...")
                # Load VAD model
                vad_model, get_speech_timestamps, read_audio = load_silero_vad()
                
                # Load audio with silero's function (works with WAV files)
                wav = read_audio(audio_file)
                
                # Get speech timestamps
                speech_timestamps = get_speech_timestamps(
                    wav, vad_model, 
                    return_seconds=True
                )
                
                if speech_timestamps:
                    print(f"Found {len(speech_timestamps)} speech segments")
                    # Create timestamp string for debugging
                    timestamps_str = " ".join([f"[{t['start']:.1f}s -> {t['end']:.1f}s]" for t in speech_timestamps[:5]])
                    print(f"First 5 speech segments: {timestamps_str}")
                    
                    # Use Whisper's timestamp feature to focus on speech segments
                    result = pipe(audio_file, return_timestamps=True)
                else:
                    print("No speech segments detected, falling back to regular transcription")
                    result = pipe(audio_file)
            except Exception as vad_error:
                print(f"Error in VAD processing: {vad_error}. Falling back to regular transcription.")
                result = pipe(audio_file)
        else:
            # Standard transcription without VAD
            result = pipe(audio_file)
        
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio file {audio_file}: {e}")
        return None

def find_audio_files(input_dir, file_pattern="*.wav"):
    """
    Find all wav audio files in the directory
    """
    return list(Path(input_dir).glob(file_pattern))

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe WAV audio files")
    parser.add_argument("--input_dir", required=True, help="Directory containing WAV audio files")
    parser.add_argument("--output_dir", required=True, help="Directory to save transcription text files")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2 for faster inference")
    parser.add_argument("--chunk_length", type=int, default=30, help="Length of chunks in seconds for processing long audio (0 for no chunking)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing chunks")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for no limit)")
    parser.add_argument("--use_vad", action="store_true", help="Use Silero VAD for speech detection to improve transcription quality")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if GPU is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, dtype: {torch_dtype}")
    print(f"Transformers version: {transformers.__version__}")
    
    # Find all WAV files
    audio_files = find_audio_files(args.input_dir)
    print(f"Found {len(audio_files)} WAV files")
    
    # Print VAD status
    if args.use_vad:
        print("Voice Activity Detection (VAD) enabled - this will help reduce hallucinations in quiet sections")
    
    # Limit the number of files if specified
    if args.limit > 0 and args.limit < len(audio_files):
        audio_files = audio_files[:args.limit]
        print(f"Limiting processing to first {args.limit} files")
    
    # Print model information
    model_name = "openai/whisper-large-v3-turbo"
    print(f"Using model: {model_name}")
    if args.use_flash_attention:
        print("Using Flash Attention 2")
    if args.chunk_length > 0:
        print(f"Using chunked processing with chunk length: {args.chunk_length}s and batch size: {args.batch_size}")
    
    # Process each audio file
    total_time = 0
    total_audio_length = 0
    successful_files = 0
    
    for audio_file in audio_files:
        stem = audio_file.stem
        print(f"\nProcessing: {stem}")
        
        start_time = time.time()
        
        # Get audio length
        audio_length = get_audio_length(audio_file)
        if audio_length is not None:
            print(f"Audio length: {audio_length:.2f} seconds")
            total_audio_length += audio_length
        else:
            print("Could not determine audio length")
        
        # Transcribe audio
        print(f"Transcribing {audio_file} using whisper-large-v3-turbo")
        
        # Create pipeline parameters
        pipe_params = {
            "audio_file": str(audio_file),
            "model_name": model_name,
            "device": device,
            "torch_dtype": torch_dtype,
            "use_flash_attention": args.use_flash_attention,
            "use_vad": args.use_vad
        }
        
        # Add chunking parameters if specified
        if args.chunk_length > 0:
            pipe_params["chunk_length_s"] = args.chunk_length
            pipe_params["batch_size"] = args.batch_size
            
        transcription = transcribe_audio_with_whisper(**pipe_params)
        
        if transcription is None:
            print(f"Skipping {stem} due to transcription error")
            continue
        
        # Save transcription
        output_transcript = os.path.join(args.output_dir, f"{stem}.txt")
        with open(output_transcript, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        processing_time = time.time() - start_time
        total_time += processing_time
        successful_files += 1
        
        print(f"Transcription saved to {output_transcript}")
        print(f"Processing time: {processing_time:.2f} seconds")
        
        # Calculate real-time factor if audio length is available
        if audio_length is not None and audio_length > 0:
            rtf = processing_time / audio_length
            print(f"Real-time factor (RTF): {rtf:.2f}x")
    
    # Print summary
    if successful_files > 0:
        print(f"\nSuccessfully transcribed {successful_files} of {len(audio_files)} files")
        print(f"Total processing time: {total_time:.2f} seconds")
        if total_audio_length > 0:
            overall_rtf = total_time / total_audio_length
            print(f"Overall audio length: {total_audio_length:.2f} seconds")
            print(f"Overall real-time factor (RTF): {overall_rtf:.2f}x")
    else:
        print("No files were successfully processed")

if __name__ == "__main__":
    main()