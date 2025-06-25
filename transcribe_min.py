import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
# Import the class we need to patch for the bug fix
from transformers.generation.logits_process import WhisperNoSpeechDetection
from datasets import load_dataset, Audio
import warnings

# Suppress a specific UserWarning from the transformers library regarding gradient checkpointing.
# This is often seen when loading models and is not critical for this inference example.
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")

# --- MONKEY-PATCH FOR TRANSFORMERS BUG ---
# The error "TypeError: ... prepare_inputs_for_generation() missing 1 required positional argument: 'input_ids'"
# is caused by a bug in how the WhisperNoSpeechDetection processor prepares inputs for its internal forward pass.
# It calls `model.prepare_inputs_for_generation`, which is unnecessary for this specific check and causes an
# argument mismatch with some versions of the library.
#
# This patch replaces the entire `set_inputs` method. Instead of calling the problematic
# `prepare_inputs_for_generation`, it simply renames the input keys to match what the model's
# `forward` method expects ('input_features', 'decoder_input_ids') and stores them directly.
# This bypasses the error while ensuring the no-speech detection logic works correctly.

def _fixed_set_inputs(self, inputs):
    # Rename the problematic keys to what the model's forward method expects
    if "inputs" in inputs:
        inputs["input_features"] = inputs.pop("inputs")
    if "input_ids" in inputs:
        inputs["decoder_input_ids"] = inputs.pop("input_ids")
    
    # Bypass the problematic `prepare_inputs_for_generation` call by setting the inputs directly.
    self.inputs = inputs

# Apply the patch
WhisperNoSpeechDetection.set_inputs = _fixed_set_inputs
# -----------------------------------------

def main():
    """
    Minimal example of processing a long audio file using whisper-large-v3-turbo
    with specific fallback parameters for robust transcription.
    """
    # 1. Setup: Load Model and Processor
    # -------------------------------------
    # Check for GPU availability and set the device accordingly. Using a GPU is highly recommended
    # for a model of this size. On a CPU, this will be extremely slow.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    print(f"Loading model '{model_id}' on device '{device}'...")

    # Load the pre-trained model from Hugging Face.
    # - `torch_dtype` is set to float16 for GPU to save memory and improve speed.
    # - `low_cpu_mem_usage=True` prevents loading the entire model into CPU RAM first.
    # - `use_safetensors=True` is the recommended, secure way to load model weights.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    # Load the processor, which includes the feature extractor and tokenizer.
    # The processor prepares the audio data for the model and decodes the output tokens back to text.
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Audio Input: Load and Prepare
    # ---------------------------------
    # We will use a long audio sample from the Hugging Face datasets library.
    # The 'distil-whisper/meanwhile' dataset contains a suitable example.
    # This demonstrates the long-form transcription capabilities where the special parameters are used.
    print("Loading audio sample...")
    dataset = load_dataset("distil-whisper/meanwhile", "default", split="test")
    
    # Resample the audio to 16,000 Hz, which is the sampling rate Whisper expects.
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    sample = dataset[0]["audio"] # Take the first audio file

    # Process the audio array and explicitly request the attention_mask.
    # This mask is crucial for the model to distinguish real audio from padding
    # in long-form transcriptions and resolves the warning in the traceback.
    processed_inputs = processor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
        return_tensors="pt",
        return_attention_mask=True, # Ensure attention mask is returned
    )

    # Move the input features and attention mask to the same device as the model.
    input_features = processed_inputs.input_features.to(device, dtype=torch_dtype)
    attention_mask = processed_inputs.attention_mask.to(device)

    # 3. Generation with Fallback Parameters
    # --------------------------------------
    # The parameters you specified (`compression_ratio_threshold`, `logprob_threshold`,
    # `no_speech_threshold`) are part of Whisper's robustness mechanism for transcribing
    # long, noisy, or difficult audio. They control when the model should "fallback"
    # to a higher temperature (more randomness) to avoid repetitive or nonsensical output.

    print("Generating transcription with specified fallback parameters...")
    generated_ids = model.generate(
        input_features,
        attention_mask=attention_mask, # Pass the attention mask to the generate method.
        return_timestamps=True,
        # --- Your Core Parameters ---
        # If the compression ratio of the text is too high (suggesting repetition),
        # the model will retry with a higher temperature. A common value is 2.4.
        compression_ratio_threshold=2.2,

        # If the average log probability of the generated tokens is below this threshold,
        # it's considered a low-confidence transcription, triggering a fallback.
        # A common value is -1.0.
        logprob_threshold=0.5,

        # If the no-speech probability is higher than this threshold AND the logprob is
        # below `logprob_threshold`, the segment is considered silent and skipped.
        # A common value is 0.6.
        no_speech_threshold=0.5,

        # --- Required Supporting Parameters ---
        # A tuple of temperatures to try during fallback. The model starts with the first
        # temperature and moves to the next one if a fallback is triggered.
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        condition_on_prev_tokens=True, # Helps maintain context between audio chunks.
    )

    # 4. Decode and Print Output
    # --------------------------
    # Decode the generated token IDs into text.
    # `skip_special_tokens=True` removes tokens like <|startoftranscript|>, etc.
    # The output is a list of transcriptions; we take the first one.
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n--- Transcription Result ---")
    print(transcription)
    print("--------------------------\n")

if __name__ == "__main__":
    main()