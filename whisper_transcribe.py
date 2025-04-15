import os
import whisper
from praatio import textgrid
import jiwer
import argparse
import torch
import re

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
    """
    tg = textgrid.openTextgrid(textgrid_file, includeEmptyIntervals=True)
    
    # Get the specified tier
    tier = tg.getTier(tier_name)
    
    # Concatenate all non-empty labels
    reference_text = ""
    for interval in tier.entries:
        if interval.label.strip():  # Only include non-empty labels
            reference_text += " " + interval.label.strip()
    
    # Clean the concatenated text
    reference_text = clean_transcription(reference_text)
    
    return reference_text

def transcribe_audio(audio_file, model_name="base"):
    """
    Transcribe audio using Whisper model
    """
    # Load model
    model = whisper.load_model(model_name)
    
    # Transcribe
    result = model.transcribe(audio_file)
    
    return result["text"]

import jiwer

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


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper and compare with TextGrid")
    parser.add_argument("--audio", required=True, help="Path to the audio file")
    parser.add_argument("--textgrid", required=True, help="Path to the reference TextGrid file")
    parser.add_argument("--tier", default="Doctor", help="Name of the tier in TextGrid (default: Doctor)")
    parser.add_argument("--model", default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--output", default="transcription.txt", help="Path to save whisper transcription")
    
    args = parser.parse_args()
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get reference transcription from TextGrid
    print("Extracting reference transcription from TextGrid...")
    reference = extract_reference_from_textgrid(args.textgrid, args.tier)
    
    # Transcribe audio using Whisper
    print(f"Transcribing audio using Whisper {args.model} model...")
    hypothesis = transcribe_audio(args.audio, args.model)
    
    # Save transcription to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(hypothesis)
    print(f"Transcription saved to {args.output}")
    
    # Clean hypothesis text
    hypothesis_clean = clean_transcription(hypothesis)
    
    # Print sample of both texts for verification
    print("\nReference sample (first 100 chars):")
    print(reference[:100] + "...")
    print("\nHypothesis sample (first 100 chars):")
    print(hypothesis_clean[:100] + "...")
    
    # Calculate WER
    wer, measures = calculate_wer(reference, hypothesis_clean)
    print(f"\nWord Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"Insertions: {measures['insertions']}")
    print(f"Deletions: {measures['deletions']}")
    print(f"Substitutions: {measures['substitutions']}")

    # Calculate and print the total word count of the reference
    reference_word_count = len(reference.split())
    print(f"Word Count (reference): {reference_word_count}")

if __name__ == "__main__":
    main()