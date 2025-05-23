#!/usr/bin/env python3
"""
postprocess_then_summarize.py
────────────────────────────────────────────────────────────
Default:            both post-processing *and* summarisation are done
                    with one local Hugging Face model.

Optional:           supply --or_model to run post-processing on OpenRouter
                    and keep summarisation local.

Flags
─────
    --skip_post     Skip Step 1 entirely.  Assumes *.post.txt exist in
                    --post_dir and generates summaries only.

    --or_model      OpenRouter model slug for Step 1.  If omitted, the HF
                    model is used for Step 1 as well.

Other features
──────────────
  – Prompt truncation so prompt + gen ≤ model context
  – Deterministic decoding by default (temperature 0)
  – Selectable dtype (auto / fp16 / bf16 / fp32)
  – SDPA flash/mem-efficient kernels disabled (Gemma CUDA bug)

© 2025 – MIT License
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
from string import Template
from typing import Dict, Literal

import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from openai import OpenAI, OpenAIError           # pip install openai==1.*
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    GenerationConfig,
)
from huggingface_hub import snapshot_download
from rich.progress import Progress


# ╭────────────────────────── model maps ──────────────────────────╮
OR_MODELS: Dict[str, str] = {
    "qwen2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "qwen3-8b":    "qwen/qwen3-8b",
    "qwen3-14b":   "qwen/qwen3-14b",
    "qwen3-32b":   "qwen/qwen3-32b",
    "qwen3-a3b":   "qwen/qwen3-30b-a3b",
    "qwq-32b":     "qwen/qwq-32b",
    "gemma-27b":   "google/gemma-3-27b-it",
    "deepseek":    "deepseek/deepseek-chat-v3-0324",
}
HF_MODELS: Dict[str, str] = {
    "medgemma-27b": "google/medgemma-27b-text-it",
    "medgemma-4b":  "google/medgemma-4b-it",
}
HF_CTX: Dict[str, int] = {m: 8192 for m in HF_MODELS}  # same ctx for both
# ╰───────────────────────────────────────────────────────────────╯

_PIPE: TextGenerationPipeline | None = None
_PIPE_REPO: str | None = None


def _select_dtype(kind: Literal["auto", "fp16", "bf16", "fp32"], gpu: bool) -> torch.dtype:
    if kind == "fp32":
        return torch.float32
    if kind == "bf16":
        return torch.bfloat16
    if kind == "fp16":
        return torch.float16
    return torch.bfloat16 if gpu and torch.cuda.is_bf16_supported() else (
        torch.float16 if gpu else torch.float32
    )


# ───────────────────────────── Hugging Face ─────────────────────────────
def get_hf_pipeline(repo_id: str, dtype_choice: str) -> TextGenerationPipeline:
    global _PIPE, _PIPE_REPO
    if _PIPE and _PIPE_REPO == repo_id:
        return _PIPE

    local = snapshot_download(repo_id, resume_download=True,
                              token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    gpu = torch.cuda.is_available()
    dtype = _select_dtype(dtype_choice, gpu)

    model = AutoModelForCausalLM.from_pretrained(
        local, torch_dtype=dtype, device_map="auto" if gpu else None
    )
    tok = AutoTokenizer.from_pretrained(local)
    _PIPE = TextGenerationPipeline(model=model, tokenizer=tok)
    _PIPE_REPO = repo_id
    return _PIPE


def hf_generate(pipe: TextGenerationPipeline, prompt: str, *,
                max_tokens: int, temperature: float, ctx_limit: int,
                retries: int = 3, backoff: float = 2.0) -> str:
    ids = pipe.tokenizer(prompt, return_tensors="pt",
                         add_special_tokens=False).input_ids[0]
    if ids.numel() + max_tokens > ctx_limit:
        prompt = pipe.tokenizer.decode(ids[-(ctx_limit - max_tokens):],
                                       skip_special_tokens=False)

    cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        return_dict_in_generate=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    for attempt in range(retries + 1):
        try:
            txt = pipe(prompt, generation_config=cfg,
                       return_full_text=False)[0]["generated_text"]
            return txt.strip()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)


# ───────────────────────────── OpenRouter ─────────────────────────────
def get_or_client() -> OpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY missing")
    return OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": "https://example.com", "X-Title": "postprocess-script"},
        timeout=120,
    )


def or_generate(model: str, prompt: str, *,
                max_tokens: int, temperature: float,
                retries: int = 3, backoff: float = 2.0) -> str:
    client = get_or_client()
    for attempt in range(retries + 1):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return r.choices[0].message.content.strip()
        except OpenAIError:
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)
# ────────────────────────────────────────────────────────────────────────


def render(tmpl: str, text: str) -> str:
    return Template(tmpl).safe_substitute(input=text)


# ───────────────────────────────────────── main ──────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Post-process then summarise (HF by default).")
    ap.add_argument("--hf_model", required=True, choices=HF_MODELS.keys(),
                    help="HuggingFace model id to use (both steps by default).")
    ap.add_argument("--or_model", choices=OR_MODELS.keys(),
                    help="OpenRouter model for post-processing only.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--post_dir", required=True)
    ap.add_argument("--summary_dir", required=True)
    ap.add_argument("--post_prompt", required=True)
    ap.add_argument("--summary_prompt", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"],
                    default="auto")
    ap.add_argument("--skip_post", action="store_true",
                    help="Skip Step 1; use existing *.post.txt in --post_dir.")
    args = ap.parse_args()

    in_dir   = Path(args.input_dir).expanduser()
    post_dir = Path(args.post_dir).expanduser(); post_dir.mkdir(parents=True, exist_ok=True)
    sum_dir  = Path(args.summary_dir).expanduser(); sum_dir.mkdir(parents=True, exist_ok=True)

    post_tmpl = Path(args.post_prompt).read_text()
    sum_tmpl  = Path(args.summary_prompt).read_text()

    # HF pipeline
    repo_id = HF_MODELS[args.hf_model]
    ctx_max = HF_CTX[args.hf_model]
    pipe = get_hf_pipeline(repo_id, dtype_choice=args.dtype)

    # decide file list
    if args.skip_post:
        files = sorted(post_dir.glob("*.post.txt"))
        if not files:
            sys.exit("--skip_post but no *.post.txt in post_dir")
    else:
        files = sorted(in_dir.glob("*.txt"))
        if not files:
            sys.exit("No *.txt in input_dir")

    use_openrouter = (args.or_model is not None) and (not args.skip_post)

    with Progress() as bar:
        task = bar.add_task("[green]Processing…", total=len(files))
        for fp in files:
            if args.skip_post:
                processed = fp.read_text()
                base = fp.stem.split(".post", 1)[0]
            else:
                raw = fp.read_text()
                if use_openrouter:
                    processed = or_generate(
                        OR_MODELS[args.or_model],
                        render(post_tmpl, raw),
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                else:
                    processed = hf_generate(
                        pipe,
                        render(post_tmpl, raw),
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        ctx_limit=ctx_max,
                    )
                (post_dir / f"{fp.stem}.post.txt").write_text(processed)
                base = fp.stem

            summary = hf_generate(
                pipe,
                render(sum_tmpl, processed),
                max_tokens=args.max_tokens // 2,
                temperature=args.temperature,
                ctx_limit=ctx_max,
            )
            (sum_dir / f"{base}.sum.txt").write_text(summary)
            bar.advance(task)

    print(f"✅ Done. Post-files → {post_dir} | Summaries → {sum_dir}")


if __name__ == "__main__":
    main()
