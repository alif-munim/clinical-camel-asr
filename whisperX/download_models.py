import os  
import torch  
from pathlib import Path  
  
# Set ALL HuggingFace cache directories BEFORE any imports  
os.environ['HF_HOME'] = './models/huggingface'  
os.environ['HUGGINGFACE_HUB_CACHE'] = './models/huggingface/hub'  
os.environ['HF_HUB_CACHE'] = './models/huggingface/hub'  # Alternative cache var  
os.environ['TRANSFORMERS_CACHE'] = './models/huggingface/transformers'  
os.environ['TORCH_HOME'] = './models/torch'  
  
# Import after setting environment variables  
import whisperx  
from whisperx.diarize import DiarizationPipeline  
  
def download_all_models():  
    """Download all WhisperX models to local directories for offline usage."""  
      
    # Create directories with full structure  
    os.makedirs('./models/whisper', exist_ok=True)  
    os.makedirs('./models/alignment', exist_ok=True)  
    os.makedirs('./models/huggingface/hub', exist_ok=True)  
    os.makedirs('./models/huggingface/transformers', exist_ok=True)  
    os.makedirs('./models/torch', exist_ok=True)  
      
    # Read HF_TOKEN from environment  
    HF_TOKEN = os.getenv('HF_TOKEN')  
    if not HF_TOKEN:  
        print("Warning: HF_TOKEN not found in environment variables.")  
        print("Set HF_TOKEN environment variable to download diarization models.")  
        return  
      
    print("Starting model downloads...")  
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")  
    print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE')}")  
      
    # Download Whisper model  
    print("Downloading Whisper model...")  
    model = whisperx.load_model(  
        "large-v3",   
        device="cuda",   
        download_root="./models/whisper",  
        vad_method="silero"  
    )  
    print("✓ Whisper model downloaded")  
      
    # Download alignment model  
    print("Downloading alignment model...")  
    align_model, metadata = whisperx.load_align_model(  
        language_code="en",   
        device="cuda",  
        model_dir="./models/alignment"  
    )  
    print("✓ Alignment model downloaded")  
      
    # Download diarization model with explicit cache control  
    print("Downloading diarization model...")  
    try:  
        # Force the cache directory one more time right before creating the pipeline  
        import huggingface_hub  
        huggingface_hub.constants.HF_HUB_CACHE = Path("./models/huggingface/hub").resolve()  
          
        diarize_model = DiarizationPipeline(  
            use_auth_token=HF_TOKEN,  
            device="cuda"  
        )  
        print("✓ Diarization model downloaded")  
          
        # Verify where it actually downloaded  
        local_hf_dir = Path("./models/huggingface/hub")  
        if local_hf_dir.exists():  
            diarization_models = list(local_hf_dir.glob("models--pyannote--speaker-diarization*"))  
            if diarization_models:  
                print(f"✓ Diarization model found in local cache: {diarization_models[0]}")  
            else:  
                print("⚠ Diarization model may have downloaded to default cache")  
                # Copy from default location if needed  
                default_cache = Path.home() / ".cache" / "huggingface" / "hub"  
                default_models = list(default_cache.glob("models--pyannote--speaker-diarization*"))  
                if default_models:  
                    print(f"Copying from default cache: {default_models[0]}")  
                    import shutil  
                    shutil.copytree(default_models[0], local_hf_dir / default_models[0].name, dirs_exist_ok=True)  
                    print("✓ Diarization model copied to local directory")  
          
    except Exception as e:  
        print(f"✗ Failed to download diarization model: {e}")  
      
    # Check final locations  
    check_model_paths()  
  
def check_model_paths():  
    """Check and print actual paths of downloaded models."""  
    print("\nFinal model locations:")  
      
    # Check Silero VAD with broader file search  
    torch_dir = Path("./models/torch/hub")  
    if torch_dir.exists():  
        # Look for any files, not just specific extensions  
        torch_files = [f for f in torch_dir.rglob("*") if f.is_file()]  
        print(f"\n📁 Silero VAD in {torch_dir}:")  
        if torch_files:  
            # Show model-related files  
            model_files = [f for f in torch_files if any(ext in f.name.lower() for ext in ['model', '.pth', '.pt', '.bin', '.onnx'])]  
            for f in model_files[:5]:  # Show first 5 model files  
                size = f.stat().st_size / (1024**2)  # MB  
                print(f"  - {f.relative_to(torch_dir)} ({size:.1f} MB)")  
            if not model_files:  
                print(f"  - Found {len(torch_files)} files (no obvious model files)")  
        else:  
            print("  - No files found")

if __name__ == "__main__":  
    download_all_models()