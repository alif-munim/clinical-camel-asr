import os
import whisper
from praatio import textgrid
import jiwer
import argparse
import torch
import re
import pandas as pd
from pathlib import Path
import csv
import time

def clean_transcription(text):
    """
    Cleans transcription text by:
    1. Removing <UNIN/>, <UNSURE>text</UNSURE>, etc.
    2. Removing extra whitespace
    3. Converting to lowercase
    """
    # Remove <UNIN/> tags
    text = re.sub(r'<UNIN/>', '', text)
    
    # Remove <UNSURE>text</UNSURE> tags but keep the text within
    text = re.sub(r'<UNSURE>(.*?)</UNSURE>', r'\1', text)
    
    # Remove any other tags that might be present
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace and lowercase
    text = ' '.join(text.split()).lower()
    
    return text

def extract_reference_from_textgrid(textgrid_file, tier_name="Doctor"):
    """
    Extracts and concatenates all non-empty labels from the specified tier.
    If the specified tier is not found, tries to automatically identify the correct tier.
    """
    try:
        tg = textgrid.openTextgrid(textgrid_file, includeEmptyIntervals=True)
        
        # Print available tiers for debugging
        available_tiers = tg.tierNames
        print(f"Available tiers in {os.path.basename(textgrid_file)}: {available_tiers}")
        
        # First attempt to get the specified tier
        if tier_name in available_tiers:
            tier = tg.getTier(tier_name)
        # If patient file, try "Patient" tier
        elif "patient" in os.path.basename(textgrid_file).lower() and "Patient" in available_tiers:
            print(f"Using 'Patient' tier instead of '{tier_name}' for patient file")
            tier = tg.getTier("Patient")
        # If multiple tiers are available, use the first one
        elif len(available_tiers) > 0:
            alternative_tier = available_tiers[0]
            print(f"Using alternative tier '{alternative_tier}' instead of '{tier_name}'")
            tier = tg.getTier(alternative_tier)
        else:
            raise ValueError(f"No valid tiers found in {textgrid_file}")
        
        # Concatenate all non-empty labels
        reference_text = ""
        for interval in tier.entries:
            if interval.label.strip():  # Only include non-empty labels
                reference_text += " " + interval.label.strip()
        
        # Clean the concatenated text
        reference_text = clean_transcription(reference_text)
        
        return reference_text
    except Exception as e:
        print(f"Error processing TextGrid file {textgrid_file}: {e}")
        return None

def transcribe_audio(audio_file, model_name="base"):
    """
    Transcribe audio using Whisper model
    """
    try:
        # Load model
        model = whisper.load_model(model_name)
        
        # Transcribe
        result = model.transcribe(audio_file)
        
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

def find_matching_files(audio_dir, textgrid_dir, file_pattern="*.wav"):
    """
    Find all matching audio and textgrid files
    """
    audio_files = list(Path(audio_dir).glob(file_pattern))
    pairs = []
    
    for audio_file in audio_files:
        # Get stem name (without extension)
        stem = audio_file.stem
        
        # Look for corresponding TextGrid file
        textgrid_file = Path(textgrid_dir) / f"{stem}.TextGrid"
        
        if textgrid_file.exists():
            pairs.append((str(audio_file), str(textgrid_file), stem))
        else:
            print(f"Warning: No matching TextGrid file found for {audio_file}")
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files and calculate WER")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--textgrid_dir", required=True, help="Directory containing TextGrid files")
    parser.add_argument("--output_dir", default="output", help="Directory to save transcriptions and results")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--tier", default="Doctor", help="Name of the tier in TextGrid (default: Doctor)")
    parser.add_argument("--file_pattern", default="*.wav", help="Pattern to match audio files (default: *.wav)")
    parser.add_argument("--auto_tier", action="store_true", help="Automatically determine tier based on filename")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find matching audio and TextGrid files
    file_pairs = find_matching_files(args.audio_dir, args.textgrid_dir, args.file_pattern)
    print(f"Found {len(file_pairs)} matching audio-TextGrid pairs")
    
    # Prepare results storage
    results = []
    
    # Process each file pair
    for audio_file, textgrid_file, stem in file_pairs:
        print(f"\nProcessing: {stem}")
        
        start_time = time.time()
        
        # Determine the tier to use based on filename if auto_tier is enabled
        tier_to_use = args.tier
        if args.auto_tier:
            if "patient" in stem.lower():
                tier_to_use = "Patient"
            elif "doctor" in stem.lower():
                tier_to_use = "Doctor"
            print(f"Auto-selecting tier: {tier_to_use} based on filename")
        
        # Extract reference transcription
        print(f"Extracting reference from {textgrid_file}")
        reference = extract_reference_from_textgrid(textgrid_file, tier_to_use)
        
        if reference is None:
            print(f"Skipping {stem} due to TextGrid extraction error")
            continue
        
        # Transcribe audio
        print(f"Transcribing {audio_file} using Whisper {args.model} model")
        hypothesis = transcribe_audio(audio_file, args.model)
        
        if hypothesis is None:
            print(f"Skipping {stem} due to transcription error")
            continue
        
        # Save transcription
        output_transcript = os.path.join(args.output_dir, f"{stem}.txt")
        with open(output_transcript, 'w', encoding='utf-8') as f:
            f.write(hypothesis)
        print(f"Transcription saved to {output_transcript}")
        
        # Clean hypothesis text
        hypothesis_clean = clean_transcription(hypothesis)
        
        # Calculate WER
        wer, measures = calculate_wer(reference, hypothesis_clean)
        processing_time = time.time() - start_time
        
        # Store results
        result = {
            'file': stem,
            'wer': wer,
            'insertions': measures['insertions'],
            'deletions': measures['deletions'],
            'substitutions': measures['substitutions'],
            'ref_word_count': len(reference.split()),
            'processing_time': processing_time
        }
        results.append(result)
        
        # Print results for this file
        print(f"Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
        print(f"Insertions: {measures['insertions']}")
        print(f"Deletions: {measures['deletions']}")
        print(f"Substitutions: {measures['substitutions']}")
        print(f"Word Count (reference): {result['ref_word_count']}")
        print(f"Processing time: {processing_time:.2f} seconds")
    
    # If we have results, save to CSV
    if results:
        # Save detailed results to CSV
        results_df = pd.DataFrame(results)
        csv_file = os.path.join(args.output_dir, "wer_results.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"\nDetailed results saved to {csv_file}")
        
        # Print summary
        avg_wer = results_df['wer'].mean()
        print(f"\nSummary of {len(results)} files:")
        print(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print(f"Best WER: {results_df['wer'].min():.4f} ({results_df['wer'].min()*100:.2f}%)")
        print(f"Worst WER: {results_df['wer'].max():.4f} ({results_df['wer'].max()*100:.2f}%)")
        
        # Create a summary file
        summary_file = os.path.join(args.output_dir, "wer_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Processed {len(results)} files using Whisper {args.model} model\n")
            f.write(f"Average WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)\n")
            f.write(f"Best WER: {results_df['wer'].min():.4f} ({results_df['wer'].min()*100:.2f}%)\n")
            f.write(f"Worst WER: {results_df['wer'].max():.4f} ({results_df['wer'].max()*100:.2f}%)\n")
            
            # List files sorted by WER
            f.write("\nFiles sorted by WER (best to worst):\n")
            for _, row in results_df.sort_values('wer').iterrows():
                f.write(f"{row['file']}: {row['wer']:.4f} ({row['wer']*100:.2f}%)\n")
        
        print(f"Summary saved to {summary_file}")
    else:
        print("No results were successfully processed")

if __name__ == "__main__":
    main()