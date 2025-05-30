#!/usr/bin/env python3
"""
Batch post-processing + summarisation via OpenRouter.

© 2025  – MIT License
"""
from __future__ import annotations
import os, sys, time, argparse, glob, json, textwrap
from pathlib import Path
from typing import Dict, List

from openai import OpenAI, OpenAIError          # pip install openai==1.* 
from rich.progress import Progress             # pip install rich
from string import Template


# ------------------------ model map ------------------------
MODEL_ALIASES: Dict[str, str] = {
    "qwen2.5-72b":  "qwen/qwen-2.5-72b-instruct",
    "qwen3-8b": "qwen/qwen3-8b",
    "qwen3-14b": "qwen/qwen3-14b",
    "qwen3-32b": "qwen/qwen3-32b",
    "qwen3-a3b": "qwen/qwen3-30b-a3b",
    "qwq-32b":   "qwen/qwq-32b",
    "gemma-27b": "google/gemma-3-27b-it",
    "deepseek":  "deepseek/deepseek-chat-v3-0324",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "phi-4": "microsoft/phi-4",
    "gemma-3-27b": "google/gemma-3-27b-it"
}
# -----------------------------------------------------------


def get_client(api_key: str, referrer: str = "https://example.com",
               title: str = "postproc-script") -> OpenAI:
    """Return an OpenAI-SDK client configured for OpenRouter."""
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": referrer,  # required by OpenRouter TOS
            "X-Title": title
        },
        timeout=120
    )


def call_llm(client: OpenAI, model: str, prompt: str,
             max_tokens: int = 2048, temperature: float = 0.2,
             retries: int = 3, retry_backoff: float = 2.0) -> str:
    """Robust wrapper around chat.completions.create()."""
    attempt, err = 0, None
    last_exception = None
    while attempt <= retries:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return resp.choices[0].message.content.strip()
        except OpenAIError as e:
            last_exception = e
            print(f"OpenAIError on attempt {attempt + 1}: {e}", file=sys.stderr)
            # You might want to inspect e.response.text if available and relevant
            # or e.http_status, e.code etc.
            attempt += 1
            if attempt > retries:
                break # exit loop to raise last_exception
            time.sleep(retry_backoff ** attempt)
        except json.JSONDecodeError as e_json: # Catch JSONDecodeError specifically
            last_exception = e_json
            print(f"JSONDecodeError on attempt {attempt + 1}: {e_json}", file=sys.stderr)
            # This is where you'd try to get the raw response text if the OpenAI client
            # doesn't bubble it up directly in the exception.
            # Unfortunately, the error occurs *after* httpx has tried to parse .json(),
            # so accessing the raw text from the exception `e_json` itself is not direct.
            # The best way is to log it *before* the line that fails.
            # For now, we'll just print the error and retry or fail.
            # A more robust solution would involve making the request with httpx directly
            # or modifying the OpenAI client to expose the raw response on error.
            print("The API returned a non-JSON response. This often indicates a server-side error or HTML error page.", file=sys.stderr)
            attempt += 1
            if attempt > retries:
                break
            time.sleep(retry_backoff ** attempt)
        except Exception as e_generic: # Catch any other unexpected errors
            last_exception = e_generic
            print(f"Generic error on attempt {attempt + 1}: {e_generic}", file=sys.stderr)
            attempt += 1
            if attempt > retries:
                break
            time.sleep(retry_backoff ** attempt)

    if last_exception:
        raise last_exception # Re-raise the last caught exception
    # Should not be reached if retries are exhausted and an exception was caught
    raise RuntimeError("call_llm failed after multiple retries without specific exception.")

def render(template: str, text: str) -> str:
    """Insert text into template at $input while leaving other braces alone."""
    return Template(template).safe_substitute(input=text)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-process then summarise text files with OpenRouter."
    )
    ap.add_argument("--model", required=True, choices=MODEL_ALIASES.keys(),
                    help="Which model to use.")
    ap.add_argument("--input_dir", required=True, help="Folder with .txt inputs")
    ap.add_argument("--post_dir", required=True, help="Folder for post-processed outputs")
    ap.add_argument("--summary_dir", required=True, help="Folder for summaries")
    ap.add_argument("--post_prompt", required=True, help="File containing post-process prompt template")
    ap.add_argument("--summary_prompt", required=True, help="File containing summarisation prompt template")
    ap.add_argument("--max_tokens", type=int, default=2048,
                    help="max_tokens for both generations (default 2048)")
    args = ap.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Error: set OPENROUTER_API_KEY in environment")

    # resolve folders & prompts
    input_dir   = Path(args.input_dir).expanduser()
    post_dir    = Path(args.post_dir).expanduser();     post_dir.mkdir(parents=True,  exist_ok=True)
    summary_dir = Path(args.summary_dir).expanduser();  summary_dir.mkdir(parents=True, exist_ok=True)

    post_tmpl    = Path(args.post_prompt).read_text(encoding="utf-8")
    summary_tmpl = Path(args.summary_prompt).read_text(encoding="utf-8")

    model_slug = MODEL_ALIASES[args.model]
    client = get_client(api_key)

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {input_dir}")

    with Progress() as progress:
        task = progress.add_task("[green]Processing…", total=len(files))
        for fpath in files:
            raw = fpath.read_text(encoding="utf-8")
            # 1️⃣ post-processing
            pprompt = render(post_tmpl, raw)
            # print("TOKENS IN:", tokenize_len(pprompt))
            print("DEBUG prompt starts >>>", pprompt[:400], "<<<")
            print("DEBUG prompt ends >>>", pprompt[400:], "<<<")
            processed = call_llm(client, model_slug, pprompt,
                                 max_tokens=args.max_tokens)
            out_post = post_dir / f"{fpath.stem}.post.txt"
            out_post.write_text(processed, encoding="utf-8")

            # 2️⃣ summarisation
            sprompt = render(summary_tmpl, processed)
            summary = call_llm(client, model_slug, sprompt,
                               max_tokens=args.max_tokens // 2)
            out_sum = summary_dir / f"{fpath.stem}.sum.txt"
            out_sum.write_text(summary, encoding="utf-8")

            progress.advance(task)

    print(f"✅ Done. {len(files)} files post-processed → {post_dir}, "
          f"summaries → {summary_dir}")


if __name__ == "__main__":
    main()
