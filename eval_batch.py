#!/usr/bin/env python3
"""
batch_evaluate_summary.py
-------------------------
Batch-evaluate candidate summaries with the OpenAI Batch API.

Features
--------
• single evaluation  OR  many evaluations in one batch
• accepts ANY OpenAI-compatible endpoint (Azure, proxy, etc.)
• pairs transcript ↔ summary by filename when using --gt-dir/--sum-dir
• outputs one *.out.txt per job in --results-dir
• can save input prompts to a separate directory for debugging
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from time import sleep

import openai

# --------------------------------------------------------------------------- #
# basic helpers
# --------------------------------------------------------------------------- #

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def build_messages(system_prompt: str, transcript: str, cand_json: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "### ground_truth_transcript\n"
                f"{transcript}\n\n"
                "### candidate_json\n"
                f"{cand_json}"
            ),
        },
    ]

def write_jsonl(path: Path, objs: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for o in objs:
            json.dump(o, fp)
            fp.write("\n")

def save_result(out_dir: Path, job_id: str, content: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{job_id}.out.txt").write_text(content, encoding="utf-8")

def save_prompt(prompt_dir: Path, job_id: str, messages: list[dict]) -> None:
    """Save the complete prompt that's sent to the API."""
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / f"{job_id}.prompt.json"
    with prompt_file.open("w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)

# --------------------------------------------------------------------------- #
# job collectors
# --------------------------------------------------------------------------- #

def jobs_from_triple_dir(d: Path) -> list[tuple[Path, Path, Path, str]]:
    """Expect  XX_prompt.txt  XX_transcript.txt  XX_summary.json  (XX arbitrary)."""
    jobs: list[tuple[Path, Path, Path, str]] = []
    for summ in d.glob("*_summary.*"):
        stem = summ.stem.rsplit("_", 1)[0]
        prompt     = d / f"{stem}_prompt.txt"
        transcript = d / f"{stem}_transcript.txt"
        if not prompt.exists() or not transcript.exists():
            sys.exit(f"❌  Missing {prompt} or {transcript} for stem '{stem}'")
        jobs.append((prompt, transcript, summ, stem))
    if not jobs:
        sys.exit(f"❌  No *_summary.* files found in {d}")
    return jobs


def jobs_from_two_dirs(gt_dir: Path, sum_dir: Path,
                       common_prompt: Path) -> list[tuple[Path, Path, Path, str]]:
    """
    Pair transcript *.txt with summary *.sum.txt based on identical stem before '.sum'.
    """
    gt_map = {f.stem.lower(): f for f in gt_dir.glob("*.txt")}
    jobs: list[tuple[Path, Path, Path, str]] = []

    for summ in sum_dir.glob("*.sum.txt"):
        stem = Path(summ.name.replace(".sum", "")).stem.lower()
        if stem not in gt_map:
            sys.exit(f"❌  No transcript matching '{summ.name}' in {gt_dir}")
        jobs.append((common_prompt, gt_map[stem], summ, stem.replace(" ", "_")))
    if not jobs:
        sys.exit("❌  No matching *.sum.txt files found")
    return jobs


def single_job(prompt: Path, transcript: Path, summary: Path) -> list[tuple[Path, Path, Path, str]]:
    if not (prompt.exists() and transcript.exists() and summary.exists()):
        sys.exit("❌  One or more single-job paths do not exist")
    return [(prompt, transcript, summary, summary.stem)]

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Batch summary evaluation with OpenAI Batch API")
    # ----- input selection -----
    ap.add_argument("--prompt", required=True, help="System prompt .txt (shared by all jobs)")
    
    # Define mutually exclusive input modes
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", help="Directory containing *_prompt/_transcript/_summary triples")
    group.add_argument("--gt-dir", help="Directory with ground-truth transcripts (use with --sum-dir)")
    group.add_argument("--single-mode", action="store_true", help="Use single files (requires --transcript and --summary)")
    
    # Additional arguments
    ap.add_argument("--sum-dir", help="Directory with *.sum.txt summaries (requires --gt-dir)")
    ap.add_argument("--transcript", help="Single transcript file (requires --single-mode)")
    ap.add_argument("--summary", help="Single candidate summary file (requires --single-mode)")

    # ----- OpenAI / Batch settings -----
    ap.add_argument("--results-dir", required=True, help="Folder to write *.out.txt files")
    ap.add_argument("--prompts-dir", help="Folder to save input prompts for debugging")
    ap.add_argument("--model", default="gpt-4o-2024-05-13", help="Model name (default: gpt-4o-2024-05-13)")
    ap.add_argument("--api-base", default="https://api.openai.com/v1", help="API base URL")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--poll-interval", type=int, default=8, help="Polling seconds")
    args = ap.parse_args()

    # ---------- API key / endpoint ----------
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        sys.exit("❌  Set OPENAI_API_KEY environment variable")
    openai.api_base = args.api_base.rstrip("/")

    # ---------- collect jobs ----------
    prompt_path = Path(args.prompt)
    prompt_text = read(prompt_path)

    if args.input_dir:
        jobs = jobs_from_triple_dir(Path(args.input_dir))
    elif args.gt_dir and args.sum_dir:
        jobs = jobs_from_two_dirs(Path(args.gt_dir), Path(args.sum_dir), prompt_path)
    elif args.single_mode and args.transcript and args.summary:
        jobs = single_job(prompt_path, Path(args.transcript), Path(args.summary))
    else:
        sys.exit("❌  Invalid combination of arguments. See --help.")

    # ---------- build batch JSONL ----------
    batch_items = []
    # Store prompts for each job if prompts directory is specified
    prompts_by_job = {}
    
    for p_prompt, p_trans, p_sum, jid in jobs:
        try:
            transcript_text = read(p_trans)
            summary_text = read(p_sum)
            prompt_content = read(p_prompt)
            
            # Print the first 100 chars of each for debugging
            print(f"\nFile contents preview for job {jid}:")
            print(f"- Prompt ({len(prompt_content)} chars): {prompt_content[:100]}...")
            print(f"- Transcript ({len(transcript_text)} chars): {transcript_text[:100]}...")
            print(f"- Summary ({len(summary_text)} chars): {summary_text[:100]}...")
            
            messages = build_messages(prompt_content, transcript_text, summary_text)
            
            # Store the prompt for later saving if specified
            if args.prompts_dir:
                prompts_by_job[jid] = messages
                
            batch_items.append(
                {
                    "custom_id": jid,
                    "method": "POST",
                    "url": "/v1/chat/completions",  # Must match exactly with the endpoint in batch.create()
                    "body": {
                        "model": args.model,
                        "messages": messages,
                        "temperature": args.temperature,
                    },
                }
            )
            print(f"✓ Added job for: {jid}")
        except Exception as e:
            print(f"❌ Error processing job {jid}: {e}")
            # Optionally print more debugging info
            print(f"  prompt: {p_prompt} (exists: {p_prompt.exists()})")
            print(f"  transcript: {p_trans} (exists: {p_trans.exists()})")
            print(f"  summary: {p_sum} (exists: {p_sum.exists()})")
    
    if not batch_items:
        sys.exit("❌ No valid jobs found to process")

    # Save prompts to the specified directory if requested
    if args.prompts_dir:
        prompts_dir = Path(args.prompts_dir)
        for job_id, messages in prompts_by_job.items():
            save_prompt(prompts_dir, job_id, messages)
        print(f"📝 Saved {len(prompts_by_job)} input prompts to {prompts_dir.resolve()}")

    tmp_jsonl = Path(tempfile.mkstemp(suffix=".jsonl")[1])
    write_jsonl(tmp_jsonl, batch_items)
    print(f"📝  Prepared {len(batch_items)} requests → {tmp_jsonl}")

    # Optional: Print the first request for debugging
    with open(tmp_jsonl, "r", encoding="utf-8") as f:
        first_req = json.loads(f.readline())
        print(f"\nSample request structure (model: {first_req['body']['model']}):")
        print(f"- URL: {first_req['url']}")
        print(f"- Custom ID: {first_req['custom_id']}")
        print(f"- Message count: {len(first_req['body']['messages'])}")
        
    # ---------- upload JSONL as File (v1.x style) ----------
    try:
        upload = openai.files.create(
            file=open(tmp_jsonl, "rb"),
            purpose="batch",
        )
        print(f"⬆️  Uploaded file: {upload.id}")
    except Exception as e:
        sys.exit(f"❌ Failed to upload batch file: {e}")

    # ---------- create & poll Batch ----------
    try:
        batch = openai.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",  # Must include the leading slash here
            completion_window="24h",
        )
        print(f"🚀  Batch created: {batch.id}")
    except Exception as e:
        sys.exit(f"❌ Failed to create batch: {e}")

    while True:
        try:
            batch = openai.batches.retrieve(batch.id)
            print(f"⏳  Batch status: {batch.status}")
            if batch.status in {"completed", "failed", "expired", "cancelled"}:
                break
            sleep(args.poll_interval)
        except Exception as e:
            print(f"Error polling batch status: {e}")
            sleep(args.poll_interval)

    # ---------- handle non-success ----------
    if batch.status != "completed":
        print(f"❌  Batch finished with status '{batch.status}'")
        
        # 1) Per-line validator log
        if batch.error_file_id:
            try:
                print("🔎  Downloading validator error log …")
                log_bytes = openai.files.content(batch.error_file_id)
                log_txt = log_bytes.read().decode("utf-8", errors="replace")
                print("\nValidator log:\n" + "-" * 80)
                print(log_txt or "[empty log]")
                print("-" * 80)
                
                # Save error log to disk
                error_log_path = Path(args.results_dir) / "batch_error_log.txt"
                error_log_path.parent.mkdir(parents=True, exist_ok=True)
                error_log_path.write_text(log_txt, encoding="utf-8")
                print(f"Error log saved to: {error_log_path}")
            except Exception as e:
                print(f"Failed to download error log: {e}")
        
        # 2) Top-level errors (typical when *all* lines are rejected, e.g. bad model)
        elif getattr(batch, "errors", None):
            print("🔎  Top-level batch errors:")
            try:
                # 1.76.0 → .model_dump(); 1.5–1.7 → .dict()
                clean = batch.errors.model_dump() if hasattr(batch.errors, "model_dump") else batch.errors.dict()
                print(json.dumps(clean, indent=2))
            except Exception as e:
                # last-resort fallback: loop and stringify
                for err in batch.errors.data if hasattr(batch.errors, "data") else batch.errors:
                    print(f"- code: {getattr(err, 'code', '?')} message: {getattr(err, 'message', '?')}")
                print(f"(could not JSON-serialize errors object: {e})")
            
        # 3) Fallback: dump the whole batch object
        else:
            print("⚠️  No error_file_id and no .errors field. Batch object:")
            try:
                batch_json = json.dumps(batch, indent=2, default=str)
                print(batch_json)
            except Exception as e:
                print(f"Failed to serialize batch object: {e}")
                for key, value in vars(batch).items():
                    print(f"- {key}: {value}")
        
        sys.exit(1)

    # ---------- download merged results (batch.status == completed) ----------
    try:
        raw_bytes = openai.files.content(batch.output_file_id)
        merged_path = tmp_jsonl.with_suffix(".results.jsonl")
        
        # Write the bytes content to file directly
        merged_path.write_bytes(raw_bytes.read())
        print(f"⬇️  Downloaded results → {merged_path}")
    except Exception as e:
        sys.exit(f"❌  Failed to download merged results: {e}")

    # ---------- split into per-job files ----------
    out_dir = Path(args.results_dir)
    error_dir = out_dir / "_errors"  # Directory for storing error responses
    success_count = 0
    error_count = 0

    try:
        with open(merged_path, encoding="utf-8") as fp:
            for line in fp:
                obj = json.loads(line)
                job_id = obj["custom_id"]
                
                try:
                    # The correct nested path to the content
                    reply_txt = obj["response"]["body"]["choices"][0]["message"]["content"]
                    save_result(out_dir, job_id, reply_txt)
                    success_count += 1
                    print(f"✓ Processed result for job: {job_id}")
                except (KeyError, IndexError, TypeError) as e:
                    print(f"⚠️ Error processing job {job_id}: {e}")
                    # Save the problematic response for debugging
                    error_dir.mkdir(parents=True, exist_ok=True)
                    save_result(error_dir, job_id, json.dumps(obj, indent=2))
                    error_count += 1
                    
                    # Try to extract error message if it exists
                    if "response" in obj and "error" in obj["response"] and obj["response"]["error"]:
                        error_msg = str(obj["response"]["error"])
                        print(f"  Error message: {error_msg}")
                    elif "response" in obj and "body" in obj["response"] and "error" in obj["response"]["body"]:
                        error_msg = str(obj["response"]["body"]["error"])
                        print(f"  Error message: {error_msg}")
        
        print(f"✅ Results processing complete:")
        print(f"  - Successful evaluations: {success_count}")
        print(f"  - Failed evaluations: {error_count}")
        if error_count > 0:
            print(f"  - Error details saved to: {error_dir.resolve()}")
        
    except Exception as e:
        sys.exit(f"❌ Failed to process results: {e}")


if __name__ == "__main__":
    main()