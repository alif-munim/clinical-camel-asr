import os
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import jiwer
import argparse
import torch
import re
import pandas as pd
from pathlib import Path
import time
import librosa  # For audio length calculation

def clean_transcription(text):
    """
    Cleans transcription text by:
    1. Removing speaker labels (D: and P:)
    2. Removing extra whitespace
    3. Converting to lowercase
    """
    # Remove speaker labels like "D:" or "P:"
    text = re.sub(r'[DP]:\s*', '', text)
    
    # Remove extra whitespace and lowercase
    text = ' '.join(text.split()).lower()
    
    return text

def extract_reference_from_txt(txt_file):
    """
    Extracts text from transcript file and cleans it.
    """
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            reference_text = f.read()
        
        # Clean the text
        reference_text = clean_transcription(reference_text)
        
        return reference_text
    except Exception as e:
        print(f"Error processing transcript file {txt_file}: {e}")
        return None

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

def transcribe_audio_with_distil_whisper(audio_file, model_name="distil-whisper/distil-large-v3.5", device="cuda", torch_dtype=torch.float16):
    """
    Transcribe audio using Distil-Whisper model
    """
    try:
        # Load model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={"return_timestamps": True}
        )
        
        # Transcribe
        result = pipe(audio_file)
        
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio file {audio_file}: {e}")
        return None

def transcribe_audio_with_whisper(audio_file, model_name="openai/whisper-large-v3-turbo", device="cuda", torch_dtype=torch.float16, use_flash_attention=False, chunk_length_s=0, batch_size=1):
    """
    Transcribe audio using Whisper model
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

        # Transcribe
        result = pipe(audio_file)
        
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio file {audio_file}: {e}")
        return None

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate using jiwer.
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords()
    ])
    
    wer = jiwer.wer(
        reference, 
        hypothesis,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    
    # Also calculate other metrics
    measures = jiwer.compute_measures(
        reference, 
        hypothesis,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    
    return wer, measures

def find_matching_files(audio_dir, transcript_dir, audio_ext=".mp3", transcript_ext=".txt"):
    """
    Find all matching audio and transcript files
    """
    audio_files = list(Path(audio_dir).glob(f"*{audio_ext}"))
    pairs = []
    
    for audio_file in audio_files:
        # Get stem name (without extension)
        stem = audio_file.stem
        
        # Look for corresponding transcript file
        transcript_file = Path(transcript_dir) / f"{stem}{transcript_ext}"
        
        if transcript_file.exists():
            pairs.append((str(audio_file), str(transcript_file), stem))
        else:
            print(f"Warning: No matching transcript file found for {audio_file}")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files and calculate WER against text transcripts")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--transcript_dir", required=True, help="Directory containing transcript files")
    parser.add_argument("--output_dir", default="transcription_results", help="Directory to save transcriptions and results")
    parser.add_argument("--model_type", choices=["distil-large-v3.5", "whisper-large-v3-turbo", "whisper-large-v3"], 
                        default="distil-large-v3.5", help="Model type to use")
    parser.add_argument("--model", help="Specific model name (default depends on model_type)")
    parser.add_argument("--audio_ext", default=".mp3", help="Audio file extension (default: .mp3)")
    parser.add_argument("--transcript_ext", default=".txt", help="Transcript file extension (default: .txt)")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of files to process (0 for no limit)")
    parser.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2 for faster inference (whisper models only)")
    parser.add_argument("--chunk_length", type=int, default=0, help="Length of chunks in seconds for processing long audio (0 for no chunking, whisper models only)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing chunks (whisper models only)")
    
    args = parser.parse_args()
    
    # Set default model names based on model type
    if args.model is None:
        if args.model_type == "distil-large-v3.5":
            args.model = "distil-whisper/distil-large-v3.5"
        elif args.model_type == "whisper-large-v3-turbo":
            args.model = "openai/whisper-large-v3-turbo"
        else:  # whisper-large-v3
            args.model = "openai/whisper-large-v3"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if GPU is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device}, dtype: {torch_dtype}")
    print(f"Transformers version: {transformers.__version__}")
    
    # Find matching audio and transcript files
    file_pairs = find_matching_files(args.audio_dir, args.transcript_dir, args.audio_ext, args.transcript_ext)
    print(f"Found {len(file_pairs)} matching audio-transcript pairs")
    
    # Limit the number of files if specified
    if args.limit > 0 and args.limit < len(file_pairs):
        file_pairs = file_pairs[:args.limit]
        print(f"Limiting processing to first {args.limit} files")
    
    # Print model information
    print(f"Using model: {args.model} ({args.model_type})")
    if args.use_flash_attention and args.model_type != "distil-large-v3.5":
        print("Using Flash Attention 2")
    if args.chunk_length > 0 and args.model_type != "distil-large-v3.5":
        print(f"Using chunked processing with chunk length: {args.chunk_length}s and batch size: {args.batch_size}")
    
    # Prepare results storage
    results = []
    
    # Process each file pair
    for audio_file, transcript_file, stem in file_pairs:
        print(f"\nProcessing: {stem}")
        
        start_time = time.time()
        
        # Get audio length
        audio_length = get_audio_length(audio_file)
        if audio_length is not None:
            print(f"Audio length: {audio_length:.2f} seconds")
        else:
            print("Could not determine audio length")
        
        # Extract reference transcription
        print(f"Extracting reference from {transcript_file}")
        reference = extract_reference_from_txt(transcript_file)
        
        if reference is None:
            print(f"Skipping {stem} due to transcript extraction error")
            continue
        
        # Transcribe audio using selected model
        print(f"Transcribing {audio_file} using {args.model_type} model: {args.model}")
        if args.model_type == "distil-large-v3.5":
            hypothesis = transcribe_audio_with_distil_whisper(
                audio_file, 
                model_name=args.model,
                device=device,
                torch_dtype=torch_dtype
            )
        else:  # whisper-large-v3-turbo or whisper-large-v3
            # Create pipeline parameters
            pipe_params = {
                "audio_file": audio_file,
                "model_name": args.model,
                "device": device,
                "torch_dtype": torch_dtype,
                "use_flash_attention": args.use_flash_attention
            }
            
            # Add chunking parameters if specified
            if args.chunk_length > 0:
                pipe_params["chunk_length_s"] = args.chunk_length
                pipe_params["batch_size"] = args.batch_size
                
            hypothesis = transcribe_audio_with_whisper(**pipe_params)
        
        if hypothesis is None:
            print(f"Skipping {stem} due to transcription error")
            continue
        
        # Save transcription
        output_transcript = os.path.join(args.output_dir, f"{stem}_{args.model_type}.txt")
        with open(output_transcript, 'w', encoding='utf-8') as f:
            f.write(hypothesis)
        print(f"Transcription saved to {output_transcript}")
        
        # Clean hypothesis text
        hypothesis_clean = clean_transcription(hypothesis)
        
        # Calculate WER
        wer, measures = calculate_wer(reference, hypothesis_clean)
        processing_time = time.time() - start_time
        
        # Calculate real-time factor if audio length is available
        rtf = None
        if audio_length is not None and audio_length > 0:
            rtf = processing_time / audio_length
        
        # Store results
        result = {
            'file': stem,
            'model': args.model,
            'model_type': args.model_type,
            'wer': wer,
            'insertions': measures['insertions'],
            'deletions': measures['deletions'],
            'substitutions': measures['substitutions'],
            'ref_word_count': len(reference.split()),
            'processing_time': processing_time,
            'audio_length': audio_length,
            'rtf': rtf
        }
        results.append(result)
        
        # Print results for this file
        print(f"Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
        print(f"Insertions: {measures['insertions']}")
        print(f"Deletions: {measures['deletions']}")
        print(f"Substitutions: {measures['substitutions']}")
        print(f"Word Count (reference): {result['ref_word_count']}")
        print(f"Processing time: {processing_time:.2f} seconds")
        if rtf is not None:
            print(f"Real-time factor (RTF): {rtf:.2f}x")
    
    # If we have results, save to CSV
    if results:
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Save results file with model type in the filename
        csv_file = os.path.join(args.output_dir, f"{args.model_type}_wer_results.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"\nDetailed results saved to {csv_file}")
        
        # Print summary
        avg_wer = results_df['wer'].mean()
        print(f"\nSummary of {len(results)} files:")
        print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print(f"Best WER: {results_df['wer'].min():.4f} ({results_df['wer'].min()*100:.2f}%)")
        print(f"Worst WER: {results_df['wer'].max():.4f} ({results_df['wer'].max()*100:.2f}%)")
        print(f"Average processing time: {results_df['processing_time'].mean():.2f} seconds per file")
        
        # Calculate and print average audio length and RTF
        if not results_df['audio_length'].isna().all():
            avg_audio_length = results_df['audio_length'].mean()
            print(f"Average audio length: {avg_audio_length:.2f} seconds")
        
        if not results_df['rtf'].isna().all():
            avg_rtf = results_df['rtf'].mean()
            print(f"Average real-time factor (RTF): {avg_rtf:.2f}x")
        
        # Create a summary file
        summary_file = os.path.join(args.output_dir, f"{args.model_type}_wer_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Processed {len(results)} files using {args.model_type} model: {args.model}\n")
            f.write(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)\n")
            f.write(f"Best WER: {results_df['wer'].min():.4f} ({results_df['wer'].min()*100:.2f}%)\n")
            f.write(f"Worst WER: {results_df['wer'].max():.4f} ({results_df['wer'].max()*100:.2f}%)\n")
            f.write(f"Average processing time: {results_df['processing_time'].mean():.2f} seconds per file\n")
            
            # Add audio length and RTF information to summary
            if not results_df['audio_length'].isna().all():
                avg_audio_length = results_df['audio_length'].mean()
                f.write(f"Average audio length: {avg_audio_length:.2f} seconds\n")
            
            if not results_df['rtf'].isna().all():
                avg_rtf = results_df['rtf'].mean()
                f.write(f"Average real-time factor (RTF): {avg_rtf:.2f}x\n")
            
            # List files sorted by WER
            f.write("\nFiles sorted by WER (best to worst):\n")
            for _, row in results_df.sort_values('wer').iterrows():
                audio_len = f", Length: {row['audio_length']:.2f}s" if not pd.isna(row['audio_length']) else ""
                rtf = f", RTF: {row['rtf']:.2f}x" if not pd.isna(row['rtf']) else ""
                f.write(f"{row['file']}: {row['wer']:.4f} ({row['wer']*100:.2f}%), Time: {row['processing_time']:.2f}s{audio_len}{rtf}\n")
        
        print(f"Summary saved to {summary_file}")
    else:
        print("No results were successfully processed")

if __name__ == "__main__":
    main()