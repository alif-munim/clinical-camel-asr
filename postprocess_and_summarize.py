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
    "qwen-72b":  "qwen/qwen-2.5-72b-instruct",
    "qwq-32b":   "qwq/qwq-32b",
    "gemma-27b": "google/gemma-3b-27b-instruct",
    "deepseek":  "deepseek/deepseek-chat-v3-0324"
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
            err = e
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(retry_backoff ** attempt)
    raise err   # never reached


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
            print("TOKENS IN:", tokenize_len(pprompt))
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
