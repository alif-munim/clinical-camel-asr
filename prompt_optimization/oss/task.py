# task.py
import re
from typing import Any
from langsmith.schemas import Run, Example

# --- Helpers ---------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"^\s*(answer\s*:|final\s*diagnosis\s*:|diagnosis\s*:)\s*",
    flags=re.IGNORECASE,
)

def _to_text(x: Any) -> str:
    """Best-effort extraction of text from various run/Example shapes."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("content", "output", "text", "message", "final"):
            if k in x and isinstance(x[k], str):
                return x[k]
    if isinstance(x, list):
        return " ".join(str(t) for t in x)
    return str(x or "")

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces, trim common prefixes."""
    s = _PREFIX_RE.sub("", s or "")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- Evaluator -------------------------------------------------------------

def exact_match_evaluator(run: Run, example: Example) -> dict:
    """
    Exact string match after normalization.
    Score = 1.0 if normalized hypothesis == normalized ground truth, else 0.0.
    """
    try:
        # Dataset columns: inputs["input"], outputs["gt"]
        inp_raw = _to_text(example.inputs.get("input", ""))
        hyp_raw = _to_text(run.outputs)                      # model output
        ref_raw = _to_text(example.outputs.get("gt", ""))    # ground truth

        # --- DEBUG PRINTS (add 'input') ---
        print("───────────────────────────────────")
        print("INPUT (Dataset 'input'):")
        print(repr(inp_raw))
        print("\nHYPOTHESIS (Model Output):")
        print(repr(hyp_raw))
        print("\nREFERENCE (Ground Truth 'gt'):")
        print(repr(ref_raw))
        print("───────────────────────────────────")

        hyp = normalize_text(hyp_raw)
        ref = normalize_text(ref_raw)

        score = 1.0 if hyp == ref else 0.0
        comment = f"match: {hyp == ref} | pred='{hyp}' | gt='{ref}'"
        return {"key": "exact_string_match", "score": score, "comment": comment}

    except Exception as e:
        return {"key": "exact_string_match", "score": 0.0, "comment": f"error: {e}"}

# Exported evaluators list
evaluators = [exact_match_evaluator]
