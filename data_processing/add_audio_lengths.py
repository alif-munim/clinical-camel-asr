import pandas as pd
import librosa
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def get_audio_length(audio_file):
    """
    Get the length of an audio file in seconds using librosa
    """
    try:
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception as e:
        print(f"Error getting duration for {audio_file}: {e}")
        return None

def add_audio_lengths(csv_file, audio_dir, output_csv=None):
    """
    Read the WER results CSV file and add audio lengths
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get unique filenames
    filenames = df['file'].unique()
    
    # Create a dictionary to store audio lengths
    audio_lengths = {}
    
    # Process each file
    print(f"Processing {len(filenames)} audio files...")
    for filename in tqdm(filenames):
        # Construct the audio file path
        audio_file = os.path.join(audio_dir, f"{filename}.wav")
        
        # Check if the file exists
        if os.path.exists(audio_file):
            # Get audio length
            length = get_audio_length(audio_file)
            audio_lengths[filename] = length
        else:
            print(f"Warning: Audio file not found: {audio_file}")
            audio_lengths[filename] = None
    
    # Add audio lengths to the dataframe
    df['audio_length'] = df['file'].map(audio_lengths)
    
    # Calculate characters per second (transcription density)
    df['words_per_second'] = df['ref_word_count'] / df['audio_length']
    
    # Calculate transcription speed relative to audio length
    df['transcription_speed_ratio'] = df['processing_time'] / df['audio_length']
    
    # Save the updated dataframe
    if output_csv is None:
        # Create a new filename based on the original
        base_name = os.path.splitext(csv_file)[0]
        output_csv = f"{base_name}_with_lengths.csv"
    
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")
    
    return df

def analyze_results(df):
    """
    Perform analysis on the results with audio lengths
    """
    # Print some basic statistics
    print("\n----- Basic Statistics -----")
    print(f"Total files: {len(df)}")
    print(f"Average audio length: {df['audio_length'].mean():.2f} seconds")
    print(f"Average WER: {df['wer'].mean():.4f} ({df['wer'].mean()*100:.2f}%)")
    print(f"Average words per second: {df['words_per_second'].mean():.2f}")
    print(f"Average transcription speed ratio: {df['transcription_speed_ratio'].mean():.2f}x realtime")
    
    # Check if there's a correlation between audio length and WER
    wer_length_corr = df['audio_length'].corr(df['wer'])
    print(f"\nCorrelation between audio length and WER: {wer_length_corr:.4f}")
    
    # Check if there's a correlation between words per second and WER
    wps_wer_corr = df['words_per_second'].corr(df['wer'])
    print(f"Correlation between words per second and WER: {wps_wer_corr:.4f}")
    
    # Create plots directory
    plots_dir = "wer_analysis_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create scatter plot of audio length vs WER
    plt.figure(figsize=(10, 6))
    plt.scatter(df['audio_length'], df['wer'] * 100, alpha=0.7)
    plt.xlabel('Audio Length (seconds)')
    plt.ylabel('Word Error Rate (%)')
    plt.title('Audio Length vs. Word Error Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['audio_length'], df['wer'] * 100, 1)
        p = np.poly1d(z)
        plt.plot(df['audio_length'], p(df['audio_length']), "r--", alpha=0.8)
        plt.text(0.95, 0.95, f"Correlation: {wer_length_corr:.4f}", 
                 transform=plt.gca().transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'audio_length_vs_wer.png'))
    
    # Create scatter plot of words per second vs WER
    plt.figure(figsize=(10, 6))
    plt.scatter(df['words_per_second'], df['wer'] * 100, alpha=0.7)
    plt.xlabel('Words per Second')
    plt.ylabel('Word Error Rate (%)')
    plt.title('Speech Rate vs. Word Error Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if len(df) > 1:
        z = np.polyfit(df['words_per_second'], df['wer'] * 100, 1)
        p = np.poly1d(z)
        plt.plot(df['words_per_second'], p(df['words_per_second']), "r--", alpha=0.8)
        plt.text(0.95, 0.95, f"Correlation: {wps_wer_corr:.4f}", 
                 transform=plt.gca().transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'words_per_second_vs_wer.png'))
    
    # Create histogram of audio lengths
    plt.figure(figsize=(10, 6))
    plt.hist(df['audio_length'], bins=20, alpha=0.7)
    plt.xlabel('Audio Length (seconds)')
    plt.ylabel('Count')
    plt.title('Distribution of Audio Lengths')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'audio_length_distribution.png'))
    
    print(f"\nAnalysis plots saved to {plots_dir} directory")

def main():
    parser = argparse.ArgumentParser(description="Add audio file lengths to WER results CSV")
    parser.add_argument("--csv", required=True, help="Path to the WER results CSV file")
    parser.add_argument("--audio_dir", required=True, help="Directory containing the audio files")
    parser.add_argument("--output", help="Path to save the updated CSV file (optional)")
    parser.add_argument("--analyze", action="store_true", help="Perform analysis on the results")
    
    args = parser.parse_args()
    
    # Add audio lengths to the CSV
    df = add_audio_lengths(args.csv, args.audio_dir, args.output)
    
    # Perform analysis if requested
    if args.analyze:
        analyze_results(df)

if __name__ == "__main__":
    main()