#!/usr/bin/env python3
# =====================================================================
#  run_whisperx.py · 2025-07-31  (fast edition, updated)
#  • Transcribe one file **or** every *.wav in a directory
#  • Optional diarization (fixed N speakers or min/max range)
# ---------------------------------------------------------------------
#  Examples
#  --------
#  # Single file, diarize with exactly 2 speakers (default)
#  python run_whisperx.py --input audio.wav --output out
#
#  # Whole directory, diarize with 2 to 4 speakers
#  python run_whisperx.py --input wav_audio --output out --min_speakers 2 --max_speakers 4
#
#  # Whole directory, no diarization
#  python run_whisperx.py --input wav_audio --output out --no-diarize
# =====================================================================

import argparse, logging, os, gc, time
from pathlib import Path
from typing import Optional, Dict

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

# ----------------------------- CONFIG --------------------------------
WHISPER_MODEL_NAME      = "models/faster-whisper-large-v3"
DIARIZATION_MODEL_NAME  = "models/pyannote-diarization-3.1/config.yaml"
DEVICE_DEFAULT          = "cuda"
COMPUTE_TYPE_DEFAULT    = "float16"
BATCH_SIZE_DEFAULT      = 16
# ---------------------------------------------------------------------


# --------------------------- LOAD MODELS -----------------------------
def load_models(device: str,
                compute_type: str,
                need_diarization: bool):
    """Load WhisperX once.  (Diarization stays optional.)"""
    logging.info("Loading WhisperX …")
    t0 = time.perf_counter()
    asr_model = whisperx.load_model(
        WHISPER_MODEL_NAME,
        device=device,
        compute_type=compute_type,
        vad_method="silero",
    )
    logging.info("WhisperX ready (%.1fs)", time.perf_counter() - t0)

    diar_model = None
    if need_diarization:
        logging.info("Loading diarization model …")
        t0 = time.perf_counter()
        diar_model = DiarizationPipeline(
            model_name=DIARIZATION_MODEL_NAME,
            use_auth_token=os.getenv("HF_TOKEN"),   # let HF_TOKEN work if set
            device=device,
        )
        logging.info("Diarization model ready (%.1fs)", time.perf_counter() - t0)

    return asr_model, diar_model
# ---------------------------------------------------------------------


# ------------------------- PER-FILE ROUTINE --------------------------
def transcribe_file(
    wav_path: Path,
    asr_model,
    diar_model,
    out_raw: Path,
    out_diar: Optional[Path],
    batch_size: int,
    device: str,
    num_spk: Optional[int],
    min_spk: Optional[int],
    max_spk: Optional[int],
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
    audio = whisperx.load_audio(str(wav_path), sr=16_000)   # skip resample
    stats["duration_s"] = len(audio) / 16_000.0

    # ---------- ASR ----------
    t0 = time.perf_counter()
    result = asr_model.transcribe(
        audio,
        batch_size=batch_size,
        print_progress=True,
        language="en",
    )
    stats["transcribe_s"] = time.perf_counter() - t0

    # ---------- ALIGN ----------
    if result["segments"]:
        logging.info("   aligning …")
        t0 = time.perf_counter()
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
        )
        stats["align_s"] = time.perf_counter() - t0
        del align_model
        gc.collect()
        torch.cuda.empty_cache()

    # ---------- SAVE RAW ----------
    raw_text = "\n".join(seg["text"].strip() for seg in result["segments"])
    out_raw.write_text(raw_text, encoding="utf-8")

    # ---------- DIARIZATION ----------
    if diar_model and out_diar:
        logging.info("   diarizing …")
        t0 = time.perf_counter()
        # *** critical speed-up: pass the *file path*, not the waveform ***
        diar_segments = diar_model(
            str(wav_path),
            num_speakers=num_spk,
            min_speakers=min_spk,
            max_speakers=max_spk,
        )
        result = assign_word_speakers(diar_segments, result)
        stats["diarize_s"] = time.perf_counter() - t0

        diar_text = "\n".join(
            f"[{seg.get('speaker', 'UNK')}]: {seg['text'].strip()}"
            for seg in result["segments"]
        )
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
):
    """Transcribe one file or every *.wav under a directory."""
    out_raw_dir   = out_root / "raw"
    out_diar_dir  = out_root / "diarized"
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    if diarize:
        out_diar_dir.mkdir(parents=True, exist_ok=True)

    asr_model, diar_model = load_models(
        device, compute_type, diarize
    )

    wav_paths = [inp] if inp.is_file() else sorted(inp.glob("*.wav"))
    if not wav_paths:
        logging.warning("No *.wav files found at %s", inp)
        return

    for idx, wav in enumerate(wav_paths, 1):
        logging.info("[%d/%d] %s", idx, len(wav_paths), wav.name)
        stats = transcribe_file(
            wav_path   = wav,
            asr_model  = asr_model,
            diar_model = diar_model,
            out_raw    = out_raw_dir  / f"{wav.stem}.txt",
            out_diar   = (out_diar_dir / f"{wav.stem}.txt") if diarize else None,
            batch_size = batch_size,
            device     = device,
            num_spk    = num_spk,
            min_spk    = min_spk,
            max_spk    = max_spk,
        )
        logging.info(
            "   audio %.1fs | ASR %.1fs | align %.1fs | diar %.1fs",
            stats["duration_s"],
            stats["transcribe_s"],
            stats["align_s"],
            stats["diarize_s"],
        )

    # -------------- CLEAN-UP --------------
    del asr_model, diar_model
    gc.collect()
    torch.cuda.empty_cache()
# ---------------------------------------------------------------------


# ------------------------------ MAIN ---------------------------------
def main():
    p = argparse.ArgumentParser(description="Fast batch transcription with WhisperX")
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="WAV file or directory containing WAVs.")
    p.add_argument("-o", "--output", default=Path("transcriptions"), type=Path,
                   help="Output root directory (raw/ & diarized/ subdirs will be created).")

    diar_grp = p.add_mutually_exclusive_group()
    diar_grp.add_argument("--diarize", dest="diarize", default=True,
                          help="Perform speaker diarization (default).")
    diar_grp.add_argument("--no-diarize", dest="diarize", action="store_false",
                          help="Disable speaker diarization.")
    p.set_defaults(diarize=True)

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
        inp         = args.input,
        out_root    = args.output,
        diarize     = args.diarize,
        num_spk     = args.num_speakers,
        min_spk     = args.min_speakers,
        max_spk     = args.max_speakers,
        device      = args.device,
        compute_type= args.compute_type,
        batch_size  = args.batch_size,
    )


if __name__ == "__main__":
    main()