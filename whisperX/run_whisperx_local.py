#!/usr/bin/env python3  
# =====================================================================  
#  run_whisperx_local.py · 2025-07-31  (fast edition, updated for local models)  
#  • Transcribe one file **or** every *.wav in a directory  
#  • Optional diarization (fixed N speakers or min/max range)  
#  • Uses locally downloaded models for offline operation  
# ---------------------------------------------------------------------  
  
import argparse, logging, os, gc, time  
from pathlib import Path  
from typing import Optional, Dict  
  
# Set local model cache directories BEFORE importing whisperx  
os.environ['HF_HOME'] = './models/huggingface'  
os.environ['HUGGINGFACE_HUB_CACHE'] = './models/huggingface/hub'  
os.environ['TORCH_HOME'] = './models/torch'  
  
import torch  
import whisperx  
from whisperx.diarize import DiarizationPipeline, assign_word_speakers  
  
# ----------------------------- CONFIG --------------------------------  
WHISPER_MODEL_NAME      = "large-v3"  # Model name, not path  
WHISPER_DOWNLOAD_ROOT   = "./models/whisper"  
ALIGNMENT_MODEL_DIR     = "./models/alignment"  
DEVICE_DEFAULT          = "cuda"  
COMPUTE_TYPE_DEFAULT    = "float16"  
BATCH_SIZE_DEFAULT      = 16  
# ---------------------------------------------------------------------  
  
# --------------------------- LOAD MODELS -----------------------------  
def load_models(device: str,
                compute_type: str,
                need_diarization: bool,
                need_alignment: bool = True,
                *,
                vad_method: str = "pyannote",
                vad_onset: float = 0.500,
                vad_offset: float = 0.363,
                chunk_size: int = 30):
    """Load WhisperX and alignment models once. Diarization stays optional."""  
      
    logging.info("Loading WhisperX model from local cache...")  
    t0 = time.perf_counter()  
    asr_model = whisperx.load_model(
        WHISPER_MODEL_NAME,
        device=device,
        compute_type=compute_type,
        vad_method=vad_method,
        vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        download_root=WHISPER_DOWNLOAD_ROOT,
        local_files_only=True
    )
    logging.info("WhisperX ready (%.1fs)", time.perf_counter() - t0)  
  
    # Load alignment model once for reuse  
    align_model, align_metadata = None, None  
    if need_alignment:  
        logging.info("Loading alignment model from local cache...")  
        t0 = time.perf_counter()  
        align_model, align_metadata = whisperx.load_align_model(  
            language_code="en",  # Assuming English  
            device=device,  
            model_dir=ALIGNMENT_MODEL_DIR  
        )  
        logging.info("Alignment model ready (%.1fs)", time.perf_counter() - t0)  
  
    # Load diarization model if needed  
    diar_model = None  
    if need_diarization:  
        logging.info("Loading diarization model from local cache...")  
        t0 = time.perf_counter()  
        diar_model = DiarizationPipeline(  
            use_auth_token=None,  # No token needed for local models  
            device=device,  
        )  
        logging.info("Diarization model ready (%.1fs)", time.perf_counter() - t0)  
  
    return asr_model, align_model, align_metadata, diar_model  
# ---------------------------------------------------------------------  
  
# ------------------------- PER-FILE ROUTINE --------------------------  
def transcribe_file(  
    wav_path: Path,  
    asr_model,  
    align_model,  
    align_metadata,  
    diar_model,  
    out_raw: Path,  
    out_diar: Optional[Path],  
    batch_size: int,  
    device: str,  
    num_spk: Optional[int],  
    min_spk: Optional[int],  
    max_spk: Optional[int],  
    include_nonspeech_markers: bool
) -> Dict[str, float]:  
    """Transcribe (and optionally diarize) a single WAV file."""  
    stats = {  
        "file"        : wav_path.name,  
        "duration_s"  : 0.0,  
        "transcribe_s": 0.0,  
        "align_s"     : 0.0,  
        "diarize_s"   : 0.0,  
    }  
  
    # ---------- load audio ----------  
    audio = whisperx.load_audio(str(wav_path), sr=16_000)  
    stats["duration_s"] = len(audio) / 16_000.0  
  
    # ---------- ASR ----------  
    t0 = time.perf_counter()  
    result = asr_model.transcribe(    
        audio,    
        batch_size=batch_size,    
        print_progress=True,    
        language="en",  
        include_nonspeech_markers=include_nonspeech_markers,
        chunk_size=30,  # Add this parameter  
    )    
    stats["transcribe_s"] = time.perf_counter() - t0  
  
    # ---------- ALIGN (using pre-loaded model) ----------  
    if result["segments"] and align_model:
        logging.info("   aligning …")
        t0 = time.perf_counter()

        # Pass ALL segments; align() will skip non-speech internally and keep them
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
        )

        stats["align_s"] = time.perf_counter() - t0

  
    # ---------- SAVE RAW ----------
    if include_nonspeech_markers:
        raw_lines = (seg["text"].strip() for seg in result["segments"])
    else:
        raw_lines = (seg["text"].strip() for seg in result["segments"]
                    if seg.get("type") != "non-speech")
    raw_text = "\n".join(raw_lines)
    out_raw.write_text(raw_text, encoding="utf-8")
  
    # ---------- DIARIZATION ----------  
    if diar_model and out_diar:  
        logging.info("   diarizing …")  
        t0 = time.perf_counter()  
        diar_segments = diar_model(  
            str(wav_path),  
            num_speakers=num_spk,  
            min_speakers=min_spk,  
            max_speakers=max_spk,  
        )  
        result = assign_word_speakers(diar_segments, result)  
        stats["diarize_s"] = time.perf_counter() - t0  
  
        # Build diarized text (match CLI style: no prefix for non-speech; avoid [UNK])
        lines = []
        last_spk = None
        for seg in result["segments"]:
            if seg.get("type") == "non-speech":
                if include_nonspeech_markers:
                    lines.append(seg["text"].strip())          # keep [UNTRANSCRIBED]
                continue                                       # else skip it
            spk = seg.get("speaker") or last_spk or "SPEAKER_00"
            last_spk = spk
            lines.append(f"[{spk}]: {seg['text'].strip()}")
        diar_text = "\n".join(lines)
        out_diar.write_text(diar_text, encoding="utf-8")
  
    return stats  
# ---------------------------------------------------------------------  
  
# --------------------------- BATCH DRIVER ----------------------------  
def transcribe_path(  
    inp: Path,  
    out_root: Path,  
    diarize: bool,  
    num_spk: Optional[int],  
    min_spk: Optional[int],  
    max_spk: Optional[int],  
    device: str,  
    compute_type: str,  
    batch_size: int,  
    *,
    vad_method: str,
    vad_onset: float,
    vad_offset: float,
    chunk_size: int,
    include_nonspeech_markers: bool
):  
    """Transcribe one file or every *.wav under a directory."""  
    out_raw_dir   = out_root / "raw"  
    out_diar_dir  = out_root / "diarized"  
    out_raw_dir.mkdir(parents=True, exist_ok=True)  
    if diarize:  
        out_diar_dir.mkdir(parents=True, exist_ok=True)  
  
    # Load all models once  
    asr_model, align_model, align_metadata, diar_model = load_models(
        device, compute_type, diarize, need_alignment=True,
        vad_method=vad_method, vad_onset=vad_onset, vad_offset=vad_offset, chunk_size=chunk_size
    )
  
    wav_paths = [inp] if inp.is_file() else sorted(inp.glob("*.wav"))  
    if not wav_paths:  
        logging.warning("No *.wav files found at %s", inp)  
        return  
  
    for idx, wav in enumerate(wav_paths, 1):  
        logging.info("[%d/%d] %s", idx, len(wav_paths), wav.name)  
        stats = transcribe_file(  
            wav_path       = wav,  
            asr_model      = asr_model,  
            align_model    = align_model,  
            align_metadata = align_metadata,  
            diar_model     = diar_model,  
            out_raw        = out_raw_dir  / f"{wav.stem}.txt",  
            out_diar       = (out_diar_dir / f"{wav.stem}.txt") if diarize else None,  
            batch_size     = batch_size,  
            device         = device,  
            num_spk        = num_spk,  
            min_spk        = min_spk,  
            max_spk        = max_spk,  
            include_nonspeech_markers=include_nonspeech_markers,
        )  
        logging.info(  
            "   audio %.1fs | ASR %.1fs | align %.1fs | diar %.1fs",  
            stats["duration_s"],  
            stats["transcribe_s"],  
            stats["align_s"],  
            stats["diarize_s"],  
        )  
  
    # -------------- CLEAN-UP --------------  
    del asr_model, align_model, diar_model  
    gc.collect()  
    torch.cuda.empty_cache()  
# ---------------------------------------------------------------------  
  
# ------------------------------ MAIN ---------------------------------  
def main():  
    p = argparse.ArgumentParser(description="Fast batch transcription with WhisperX (offline)")  
    p.add_argument("-i", "--input", required=True, type=Path,  
                   help="WAV file or directory containing WAVs.")  
    p.add_argument("-o", "--output", default=Path("transcriptions"), type=Path,  
                   help="Output root directory (raw/ & diarized/ subdirs will be created).")  

    p.add_argument(
        "--vad_method",
        choices=["pyannote", "silero"],
        default="silero",
        help="VAD backend to use (CLI parity: silero is often less choppy)."
    )
    p.add_argument("--vad_onset", type=float, default=0.500)
    p.add_argument("--vad_offset", type=float, default=0.363)
    p.add_argument("--chunk_size", type=int, default=30)
  
    diar_grp = p.add_mutually_exclusive_group()  
    diar_grp.add_argument("--diarize", dest="diarize", action="store_true",
                      help="Perform speaker diarization (default).")
    diar_grp.add_argument("--no-diarize", dest="diarize", action="store_false",  
                          help="Disable speaker diarization.")  

    p.set_defaults(diarize=True)  

    # add near your other argparse options
    ns_grp = p.add_mutually_exclusive_group()
    ns_grp.add_argument(
        "--include-nonspeech-markers",
        dest="include_nonspeech_markers",
        action="store_true",
        help="Insert [UNTRANSCRIBED] lines where VAD detects non-speech."
    )
    ns_grp.add_argument(
        "--no-nonspeech-markers",
        dest="include_nonspeech_markers",
        action="store_false",
        help="Omit [UNTRANSCRIBED] lines in outputs (default)."
    )
    p.set_defaults(include_nonspeech_markers=False)

  
    spk = p.add_argument_group("speaker options (ignored if --no-diarize)")  
    spk.add_argument("--num_speakers",   type=int, default=2,  
                     help="Exact number of speakers (default: 2).")  
    spk.add_argument("--min_speakers",   type=int, default=None,  
                     help="Minimum number of speakers.")  
    spk.add_argument("--max_speakers",   type=int, default=None,  
                     help="Maximum number of speakers.")  
  
    p.add_argument("--device",       default=DEVICE_DEFAULT)  
    p.add_argument("--compute_type", default=COMPUTE_TYPE_DEFAULT)  
    p.add_argument("--batch_size",   default=BATCH_SIZE_DEFAULT, type=int)  
  
    args = p.parse_args()  
    logging.basicConfig(level=logging.INFO, format="%(message)s")  
  
    transcribe_path(
        inp=args.input,
        out_root=args.output,
        diarize=args.diarize,
        num_spk=args.num_speakers,
        min_spk=args.min_speakers,
        max_spk=args.max_speakers,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        vad_method=args.vad_method,
        vad_onset=args.vad_onset,
        vad_offset=args.vad_offset,
        chunk_size=args.chunk_size,
        include_nonspeech_markers=args.include_nonspeech_markers,
    )

  
if __name__ == "__main__":  
    main()
