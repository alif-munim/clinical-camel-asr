import os
import re
import pandas as pd

# --- CONFIG ---
raw_dir       = "local_run/raw"
diarized_dir  = "local_run/diarized"
gt_csv        = "gt19.csv"                # has at least column "AUDIO"
out_csv       = "gt19_with_raw_diarized.csv"

# --- helpers ---
def leading_id(s: str) -> str:
    """Extract leading digits from a filename or label."""
    base = os.path.basename(str(s))
    m = re.match(r"^(\d+)", base)
    if not m:
        # try patterns like "77_-_..." or spaces before digits
        m = re.search(r"(\d+)", base)
    return m.group(1) if m else None

def load_dir_as_id_to_text(d):
    id_to_text, id_to_path = {}, {}
    for f in os.listdir(d):
        if not f.lower().endswith(".txt"):
            continue
        _id = leading_id(f)
        if not _id:
            continue
        p = os.path.join(d, f)
        with open(p, "r", encoding="utf-8") as fh:
            id_to_text[_id] = fh.read().strip()
        id_to_path[_id] = p
    return id_to_text, id_to_path

# --- load directories ---
raw_map_text,  raw_map_path  = load_dir_as_id_to_text(raw_dir)
dia_map_text,  dia_map_path  = load_dir_as_id_to_text(diarized_dir)

# --- load GT and derive IDs from AUDIO column ---
gt = pd.read_csv(gt_csv)
if "AUDIO" not in gt.columns:
    raise ValueError("gt19.csv must contain an 'AUDIO' column.")

gt["ID"] = gt["AUDIO"].apply(leading_id)

# --- attach contents (and optionally paths) ---
gt["raw"]      = gt["ID"].map(raw_map_text)
gt["diarized"] = gt["ID"].map(dia_map_text)

# Optional: keep the source file paths for debugging (comment out if not needed)
gt["raw_path"]      = gt["ID"].map(raw_map_path)
gt["diarized_path"] = gt["ID"].map(dia_map_path)

# --- sanity checks ---
missing_raw = gt["raw"].isna().sum()
missing_dia = gt["diarized"].isna().sum()
if missing_raw or missing_dia:
    print(f"Warning: missing mappings -> raw: {missing_raw}, diarized: {missing_dia}")

# --- save ---
cols = ["AUDIO", "GT"] if "GT" in gt.columns else ["AUDIO"]
cols += ["raw", "diarized", "raw_path", "diarized_path"]
gt.to_csv(out_csv, index=False, columns=[c for c in cols if c in gt.columns])

print(f"Saved: {out_csv}")
