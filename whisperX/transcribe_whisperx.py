import whisperx
import os
import gc
import torch
from pathlib import Path
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

# NOTE: You may need to run the script once with an internet connection
# to download the alignment and the new diarization models if you don't have them cached.

def transcribe_all_files():
    # Configuration
    device = "cuda"
    compute_type = "float16"
    batch_size = 16

    # --- Local Model Paths ---
    # Use os.path.expanduser to resolve the '~' to your home directory
    home_dir = os.path.expanduser("~")
    cache_dir = Path(home_dir) / ".cache" / "huggingface" / "hub"

    # 1. Path to the Whisper model (using the Systran variant from your cache)
    whisper_model_path = cache_dir / "models--Systran--faster-whisper-large-v3"

    # 2. Path to the Alignment model (standard model for English)
    # Make sure this model is downloaded in your cache.
    align_model_path = cache_dir / "models--facebook--wav2vec2-base-960h"

    # 3. Path to a non-gated Diarization model to remove HF_TOKEN requirement
    # We are using 2.1 which does not require a token.
    diarization_model_path = "pyannote/speaker-diarization-3.1"
    
    # --- Directory Paths ---
    audio_dir = Path("wav_audio")
    raw_output_dir = Path("transcriptions_v3/raw")
    diarized_output_dir = Path("transcriptions_v3/diarized")

    # Create output directories
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    diarized_output_dir.mkdir(parents=True, exist_ok=True)

    # Get all WAV files
    wav_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files to transcribe")

    # --- Load Models from Local Paths ---

    # Load WhisperX model from local path
    print(f"Loading WhisperX model from: {whisper_model_path}")
    model = whisperx.load_model(
        "large-v3",  # Pass the local path as a string
        device=device,
        compute_type=compute_type,
        vad_method="silero"
    )

    # Load Diarization model from local path (no token needed for this model)
    print(f"Loading Diarization model from: {diarization_model_path}")
    # We specify the local path using the `model` parameter.
    diarize_model = DiarizationPipeline(
        model_name=str(diarization_model_path), # Use model_name for local path
        use_auth_token=None,
        device=device
    )

    # Process each file
    for i, audio_file in enumerate(wav_files, 1):
        print(f"\n[{i}/{len(wav_files)}] Processing: {audio_file.name}")

        try:
            # Load audio
            audio = whisperx.load_audio(str(audio_file))

            # Transcribe
            # Note: Added vad_options to the transcribe call
            result = model.transcribe(
                audio,
                batch_size=batch_size,
                print_progress=True
            )

            # Save raw transcript
            raw_text = "\n".join([segment["text"].strip() for segment in result["segments"]])
            raw_file = raw_output_dir / f"{audio_file.stem}.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            print(f"Saved raw transcript to: {raw_file}")

            # Perform alignment for better timestamps
            if len(result["segments"]) > 0:
                print("Performing alignment...")
                # Note: whisperx.load_align_model can also take a local path
                align_model, metadata = whisperx.load_align_model(
                    language_code=result["language"],
                    device=device,
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

            # Perform diarization (no longer needs HF_TOKEN check)
            try:
                print("Performing diarization...")
                diarize_segments = diarize_model(audio) # Pass the loaded audio tensor
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

            except Exception as e:
                print(f"Diarization failed for {audio_file.name}: {e}")
                print("Continuing with next file...")

        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            continue

    # Clean up main models
    del model
    del diarize_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nCompleted! Raw transcripts in '{raw_output_dir}', diarized in '{diarized_output_dir}'")


if __name__ == "__main__":
    transcribe_all_files()