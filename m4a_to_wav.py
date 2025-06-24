from pathlib import Path
from pydub import AudioSegment
import shutil

src_root  = Path("/home/bowang/Documents/alif/clinical-camel-asr/m4a_audio")     # where the .m4a files live
dest_root = Path("/home/bowang/Documents/alif/clinical-camel-asr/wav_audio") # will be created if missing

for m4a_path in src_root.rglob("*.m4a"):    # recursive; use .glob() for non-recursive
    rel_path = m4a_path.relative_to(src_root)        # keep the same sub-folder layout
    wav_path = dest_root / rel_path.with_suffix(".wav")

    wav_path.parent.mkdir(parents=True, exist_ok=True)  # make sub-dirs as needed

    # decode ➜ write 16-bit PCM WAV
    AudioSegment.from_file(m4a_path).export(wav_path, format="wav")

    print(f"✔ {m4a_path}  →  {wav_path}")
