#!/usr/bin/env python3
"""
postprocess_or_then_summarize_hf.py
────────────────────────────────────────────────────────────
• Step 1  (post-process)  : OpenRouter chat completion
• Step 2  (summarise)     : local Hugging Face model

Safety guards:
  – prompt-truncation so prompt + gen ≤ context (HF side)
  – deterministic decoding by default (temperature 0)
  – selectable dtype (auto / fp16 / bf16 / fp32)

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
torch.backends.cuda.enable_math_sdp(True)   # make sure a kernel is enabled

from openai import OpenAI, OpenAIError         # pip install openai==1.*
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    GenerationConfig,
)
from huggingface_hub import snapshot_download
from rich.progress import Progress


# ╭──────────────────────────── Model maps ───────────────────────────╮
OR_MODELS: Dict[str, str] = {   # OpenRouter slugs
    "qwen2.5-72b":  "qwen/qwen-2.5-72b-instruct",
    "qwen3-8b": "qwen/qwen3-8b",
    "qwen3-14b": "qwen/qwen3-14b",
    "qwen3-32b": "qwen/qwen3-32b",
    "qwen3-a3b": "qwen/qwen3-30b-a3b",
    "qwq-32b":   "qwen/qwq-32b",
    "gemma-27b": "google/gemma-3-27b-it",
    "deepseek":  "deepseek/deepseek-chat-v3-0324"
}
HF_MODELS: Dict[str, str] = {   # summarisation (local)
    "medgemma-27b": "google/medgemma-27b-text-it",
}
HF_CTX: Dict[str, int] = {      # context length
    "medgemma-27b": 8192,
}
# ╰───────────────────────────────────────────────────────────────────╯

# cache for HF pipeline
_PIPELINE: TextGenerationPipeline | None = None
_LOADED_REPO: str | None = None


def _select_dtype(desired: Literal["auto", "fp16", "bf16", "fp32"], gpu: bool) -> torch.dtype:
    if desired == "fp32":
        return torch.float32
    if desired == "bf16":
        return torch.bfloat16
    if desired == "fp16":
        return torch.float16
    return torch.bfloat16 if gpu and torch.cuda.is_bf16_supported() else (
        torch.float16 if gpu else torch.float32
    )


# ───────────────────────────── OpenRouter helpers ─────────────────────────────
def get_or_client(api_key: str, referrer: str = "https://example.com",
                  title: str = "postprocess-script") -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": referrer, "X-Title": title},
        timeout=120,
    )


def call_or(model: str, prompt: str, max_tokens: int,
            temperature: float, retries: int = 3, backoff: float = 2.0) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Error: set OPENROUTER_API_KEY")
    client = get_or_client(key)
    err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except OpenAIError as e:
            err = e
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)
    raise err  # unreachable
# ───────────────────────────────────────────────────────────────────────────────


# ───────────────────────────── HuggingFace helpers ────────────────────────────
def get_hf_pipeline(repo_id: str, *, dtype_choice: str, token: str | None) -> TextGenerationPipeline:
    global _PIPELINE, _LOADED_REPO
    if _PIPELINE and _LOADED_REPO == repo_id:
        return _PIPELINE

    local_dir = snapshot_download(repo_id, resume_download=True, token=token)
    gpu = torch.cuda.is_available()
    dtype = _select_dtype(dtype_choice, gpu)

    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        device_map="auto" if gpu else None,
    )
    tok = AutoTokenizer.from_pretrained(local_dir)
    _PIPELINE = TextGenerationPipeline(model=model, tokenizer=tok)
    _LOADED_REPO = repo_id
    return _PIPELINE


def hf_generate(pipe: TextGenerationPipeline, prompt: str, *,
                max_tokens: int, temperature: float, ctx_limit: int,
                retries: int = 3, backoff: float = 2.0) -> str:
    # trim prompt to fit context
    ids = pipe.tokenizer(prompt, return_tensors="pt",
                         add_special_tokens=False).input_ids[0]
    if ids.numel() + max_tokens > ctx_limit:
        keep = ctx_limit - max_tokens
        prompt = pipe.tokenizer.decode(ids[-keep:], skip_special_tokens=False)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        return_dict_in_generate=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    for attempt in range(retries + 1):
        try:
            out = pipe(prompt, generation_config=gen_cfg, return_full_text=False)[0]["generated_text"]
            return out.strip()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)
# ───────────────────────────────────────────────────────────────────────────────


def render(tmpl: str, text: str) -> str:
    return Template(tmpl).safe_substitute(input=text)


# ───────────────────────────────────────── main ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("Post-process with OpenRouter, summarise with local HF")
    ap.add_argument("--or_model", required=True, choices=OR_MODELS.keys(),
                    help="Model slug for OpenRouter post-processing.")
    ap.add_argument("--hf_model", required=True, choices=HF_MODELS.keys(),
                    help="Local HF model for summarisation.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--post_dir", required=True)
    ap.add_argument("--summary_dir", required=True)
    ap.add_argument("--post_prompt", required=True)
    ap.add_argument("--summary_prompt", required=True)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"],
                    default="auto")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser()
    post_dir = Path(args.post_dir).expanduser(); post_dir.mkdir(parents=True, exist_ok=True)
    sum_dir = Path(args.summary_dir).expanduser(); sum_dir.mkdir(parents=True, exist_ok=True)

    post_tmpl = Path(args.post_prompt).read_text()
    sum_tmpl = Path(args.summary_prompt).read_text()

    # init HF pipeline
    repo_id = HF_MODELS[args.hf_model]
    ctx_max = HF_CTX[args.hf_model]
    pipe = get_hf_pipeline(repo_id, dtype_choice=args.dtype,
                           token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

    files = sorted(in_dir.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files in {in_dir}")

    with Progress() as bar:
        task = bar.add_task("[green]Processing…", total=len(files))
        for fp in files:
            raw = fp.read_text()

            # 1️⃣ post-process via OpenRouter
            processed = call_or(
                OR_MODELS[args.or_model],
                render(post_tmpl, raw),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            (post_dir / f"{fp.stem}.post.txt").write_text(processed)

            # 2️⃣ summarise locally
            summary = hf_generate(
                pipe,
                render(sum_tmpl, processed),
                max_tokens=args.max_tokens // 2,
                temperature=args.temperature,
                ctx_limit=ctx_max,
            )
            (sum_dir / f"{fp.stem}.sum.txt").write_text(summary)
            bar.advance(task)

    print(f"✅ {len(files)} files → {post_dir}  |  summaries → {sum_dir}")


if __name__ == "__main__":
    main()
