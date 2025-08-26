import whisperx  
import os  
import gc  
import torch  
from pathlib import Path  
from whisperx.diarize import DiarizationPipeline, assign_word_speakers  # Direct import  

# Pass a single audio file, not the folder
def transcribe_all_files():  
    # Configuration  
    device = "cuda"  
    compute_type = "float16"  
    batch_size = 16  
      
    # Paths  
    audio_dir = Path("wav_audio")  
    raw_output_dir = Path("transcriptions_v3/raw")  
    diarized_output_dir = Path("transcriptions_v3/diarized")  
      
    # Create output directories  
    raw_output_dir.mkdir(parents=True, exist_ok=True)  
    diarized_output_dir.mkdir(parents=True, exist_ok=True)  
      
    # Get all WAV files  
    wav_files = list(audio_dir.glob("*.wav"))  
    print(f"Found {len(wav_files)} WAV files to transcribe")  
      
    # Load model once  
    print("Loading WhisperX model...")  
    model = whisperx.load_model(  
        "large-v3",  
        device=device,   
        compute_type=compute_type,  
        vad_method="silero"  
    )  

    # Use local model
    HF_TOKEN = os.getenv('HF_TOKEN') 
      
    # Process each file  
    for i, audio_file in enumerate(wav_files, 1):  
        print(f"\n[{i}/{len(wav_files)}] Processing: {audio_file.name}")  
          
        try:  
            # Load audio  
            audio = whisperx.load_audio(str(audio_file))  
              
            # Transcribe  
            result = model.transcribe(  
                audio,   
                batch_size=batch_size,  
                print_progress=True  
            )  
              
            # Save raw transcript (before alignment and diarization)  
            raw_text = "\n".join([segment["text"].strip() for segment in result["segments"]])  
            raw_file = raw_output_dir / f"{audio_file.stem}.txt"  
            with open(raw_file, 'w', encoding='utf-8') as f:  
                f.write(raw_text)  
            print(f"Saved raw transcript to: {raw_file}")  
              
            # Perform alignment for better timestamps  
            if len(result["segments"]) > 0:  
                print("Performing alignment...")  
                align_model, metadata = whisperx.load_align_model(  
                    language_code=result["language"],   
                    device=device  
                )  
                result = whisperx.align(  
                    result["segments"],   
                    align_model,   
                    metadata,   
                    audio,   
                    device  
                )  
                  
                # Clean up alignment model  
                del align_model  
                gc.collect()  
                torch.cuda.empty_cache()  
              
            try:  
                if HF_TOKEN:  
                    print("Performing diarization...")  
                    diarize_model = DiarizationPipeline(
                        use_auth_token=os.environ["HF_TOKEN"],
                        device=device,
                    ) 
                    diarize_segments = diarize_model(str(audio_file)
                        # num_speakers=2,
                        # min_speakers=2,
                        # max_speakers=4
                    )  
                    result = assign_word_speakers(diarize_segments, result)  
                    
                    # Save diarized transcript  
                    diarized_text = ""  
                    for segment in result["segments"]:  
                        speaker = segment.get("speaker", "UNKNOWN")  
                        text = segment["text"].strip()  
                        diarized_text += f"[{speaker}]: {text}\n"  
                    
                    diarized_file = diarized_output_dir / f"{audio_file.stem}.txt"  
                    with open(diarized_file, 'w', encoding='utf-8') as f:  
                        f.write(diarized_text)  
                    print(f"Saved diarized transcript to: {diarized_file}")  
                    
                    del diarize_model  
                    gc.collect()  
                    torch.cuda.empty_cache()  
                else:  
                    print("Warning: HF_TOKEN not set. Skipping diarization for this file.")  
                    
            except Exception as e:  
                print(f"Diarization failed for {audio_file.name}: {e}")  
                print("Continuing with next file...")
              
        except Exception as e:  
            print(f"Error processing {audio_file.name}: {e}")  
            continue  
      
    # Clean up  
    del model  
    gc.collect()  
    torch.cuda.empty_cache()  
    print(f"\nCompleted! Raw transcripts in '{raw_output_dir}', diarized in '{diarized_output_dir}'")  
  
if __name__ == "__main__":  
    transcribe_all_files()