import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.generation.logits_process import WhisperNoSpeechDetection
import warnings

# Suppress a specific UserWarning from the transformers library regarding gradient checkpointing.
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")

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

def main():
    """
    Transcribes a long local audio file using whisper-large-v3-turbo
    with the recommended pipeline approach for robust long-form transcription.
    """
    # 1. Setup: Load Model and Create Pipeline
    # -------------------------------------
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    print(f"Loading model '{model_id}' on device '{device}'...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Create the pipeline, which is the recommended way to handle transcription.
    # It manages chunking and other complexities of long-form audio automatically.
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # 2. Transcribe Local File using the Pipeline
    # ---------------------------------
    local_audio_path = "/home/bowang/Documents/alif/clinical-camel-asr/wav_audio/3-NewPatient-RadiationOnc-Audio-ChatGPT.wav"
    print(f"Loading and transcribing local audio file: {local_audio_path}...")
    print("This might take a few minutes for a 10-minute file.")

    # Call the pipeline with the audio file path.
    # - `chunk_length_s=30` is essential. It tells the pipeline to process the audio
    #   in 30-second chunks, which is required for long-form transcription.
    # - `batch_size` can speed up transcription by processing chunks in parallel.
    # - `generate_kwargs` is used to pass your specific transcription parameters.
    result = pipe(
        local_audio_path,
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=True,
        generate_kwargs={
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "compression_ratio_threshold": 2.2,
            "logprob_threshold": 0.5,
            "no_speech_threshold": 0.3,
            "condition_on_prev_tokens": True,
        }
    )
    transcription = result["text"]

    # 3. Print the Final Result
    # --------------------------
    print("\n--- Transcription Result ---")
    print(transcription)
    print("--------------------------\n")

if __name__ == "__main__":
    main()
