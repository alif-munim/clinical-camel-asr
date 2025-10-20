# test_soap_note.py
import os
import re
import json
import atexit
import pytest
from typing import List, Tuple, Dict, Any

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

# ===================== Config =====================
TRANSCRIPT_PATH = os.getenv("TRANSCRIPT_PATH", "transcripts/patient1_transcript.txt")
NOTE_PATH       = os.getenv("NOTE_PATH",       "notes/patient1_note.json")
TABLE_OUTPUT    = os.getenv("TABLE", "0") == "1"        # set TABLE=1 to print a rich table at the end
TABLE_WIDTH     = int(os.getenv("TABLE_WIDTH", "120"))  # width hint for table wrapping

# ===================== Helpers =====================
def _norm_key(s: str) -> str:
    """Normalize JSON keys: lowercase, collapse underscores/spaces."""
    return re.sub(r"[\s_]+", " ", s).strip().lower()

def parse_json_soap_note(note_content: str) -> Dict[str, str]:
    """Parse SOAP JSON into canonical sections {CC, HPI, Impression, Plan}."""
    note_data = json.loads(note_content)
    targets = {
        "chief complaint": "CC",
        "history of present illness": "HPI",
        "impression": "Impression",
        "plan": "Plan",
    }
    sections = {}
    for k, v in note_data.items():
        nk = _norm_key(k)
        if nk in targets:
            sections[targets[nk]] = " ".join(v) if isinstance(v, list) else v
    return sections

def create_section_metric(section_name: str) -> GEval:
    """Faithfulness metric per section."""
    return GEval(
        name=f"{section_name} Faithfulness",
        evaluation_steps=[
            f"Extract all medical claims and information from the {section_name} section",
            "Verify each claim against the doctor-patient transcript",
            "Identify any contradictions or unsupported claims",
            "Heavily penalize hallucinations or information not present in the transcript",
            "Award high scores for accurate, complete information that aligns with the transcript",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        threshold=0.7,
        model="gpt-5",
    )

def eval_with_reason(metric: GEval, test_case: LLMTestCase) -> Tuple[float, bool, str]:
    """
    Run evaluation and return (score_float_0_to_1, success_bool, reason_str).
    Supports both:
      - assert_test(..., return_results=True) (newer deepeval)
      - metric.measure(...) fallback (older deepeval) where reason may be on metric
    """
    try:
        ok, results = assert_test(test_case, [metric], return_results=True)  # type: ignore[arg-type]
        res = results[0]
        # Some versions may return Decimal/bool/None; coerce safely.
        score = float(res.score if getattr(res, "score", None) is not None else 0.0)
        success = bool(res.success)
        reason = str(getattr(res, "reason", "") or "")
        return score, success, reason
    except TypeError:
        # Older deepeval: run the metric directly
        score = float(metric.measure(test_case))  # returns a float in [0,1]
        success = score >= getattr(metric, "threshold", 0.0)
        reason = str(getattr(metric, "reason", "") or "")
        return score, success, reason

# ===================== Load data =====================
with open(TRANSCRIPT_PATH, "r") as f:
    transcript = f.read()

with open(NOTE_PATH, "r") as f:
    note_content = f.read()

sections = parse_json_soap_note(note_content)

# ===================== Build tests =====================
test_data: List[Tuple[str, int, LLMTestCase, GEval]] = []
case_idx = 0
for section_name in ["CC", "HPI", "Impression", "Plan"]:
    if section_name in sections:
        test_case = LLMTestCase(
            input=f"Generate {section_name} from transcript",
            actual_output=sections[section_name],
            retrieval_context=[transcript],
        )
        metric = create_section_metric(section_name)
        test_data.append((section_name, case_idx, test_case, metric))
        case_idx += 1

# ===================== Result aggregation =====================
_RESULTS: List[Dict[str, Any]] = []

def _register_result(row: Dict[str, Any]) -> None:
    _RESULTS.append(row)

def _print_summary_table() -> None:
    if not TABLE_OUTPUT or not _RESULTS:
        return

    # Compute overall pass rate
    passed = sum(1 for r in _RESULTS if r["Status"] == "PASSED")
    total  = len(_RESULTS)
    overall = f"{(passed/total*100):.1f}%"

    # Try rich; fallback to plain text if not installed
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.box import HEAVY_HEAD
        from rich.text import Text

        console = Console(width=TABLE_WIDTH, soft_wrap=True)
        table = Table(title="Test Results", box=HEAVY_HEAD, show_lines=False, expand=True)
        table.add_column("Test case")
        table.add_column("Metric")
        table.add_column("Score")
        table.add_column("Status")
        table.add_column("Overall Success Rate")

        for r in _RESULTS:
            metric_cell = Text(r["Metric"])
            score_cell = Text(
                f"{r['Score']:.1f} (threshold={r['Threshold']}, evaluation model={r['Model']}, reason={r['Reason'] or '—'})"
            )
            table.add_row(r["Test case"], metric_cell, score_cell, r["Status"], overall)

        console.print(table)
    except Exception:
        # Plain fallback (ASCII)
        def pad(s, length):
            s = str(s)
            return s if len(s) >= length else s + " " * (length - len(s))

        headers = ["Test case", "Metric", "Score", "Status", "Overall Success Rate"]
        colw = [40, 26, 46, 8, 20]
        sep = "+".join("-" * w for w in colw)

        print("\n" + " Test Results ".center(sum(colw) + 4, "="))
        print("|".join(pad(h, w) for h, w in zip(headers, colw)))
        print(sep)
        for r in _RESULTS:
            score_txt = f"{r['Score']:.1f} (threshold={r['Threshold']}, model={r['Model']}, reason={(r['Reason'] or '—')[:28]}...)"
            row = [
                r["Test case"],
                r["Metric"],
                score_txt,
                r["Status"],
                overall,
            ]
            print("|".join(pad(x, w) for x, w in zip(row, colw)))
        print("=" * (sum(colw) + 4))

atexit.register(_print_summary_table)

# ===================== Tests =====================
@pytest.mark.parametrize("section_name,case_idx,test_case,metric", test_data)
def test_soap_section(section_name, case_idx, test_case, metric):
    print(f"\nEvaluating {section_name}")
    score, success, reason = eval_with_reason(metric, test_case)

    # Per-test console lines (kept terse; full reason appears in the table)
    print(f"{section_name} Score: {score*100:.1f}/100")
    if reason:
        print(f"{section_name} Reason: {reason}")

    # Register for table
    test_case_label = f"test_soap_section[{section_name}-test_case{case_idx}-{metric.name}]"
    _register_result({
        "Test case": test_case_label,
        "Metric": metric.name,
        "Score": score,
        "Threshold": getattr(metric, "threshold", 0.0),
        "Model": getattr(metric, "model", "unknown"),
        "Reason": reason,
        "Status": "PASSED" if success else "FAILED",
    })

    assert success, (
        reason or f"{section_name} failed threshold {getattr(metric, 'threshold', 0):.2f} with score {score:.2f}"
    )
