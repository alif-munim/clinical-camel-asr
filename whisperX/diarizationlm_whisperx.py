#!/usr/bin/env python3
"""
Run DiarizationLM on diarized text where lines look like:
[SPEAKER_00]: Hello...
[SPEAKER_01]: Hi...

- Converts to the model's expected tag style: <speaker:1> ... , <speaker:2> ...
- Preserves original line boundaries.
- Chunks prompts so that (chunk + " --> ") <= 6000 characters (per model card).
- Generates completions per chunk and transfers them back onto the hypothesis text.

Usage:
  uv run python diarizationlm_run.py --input-file path/to/diarized.txt --output-file out.txt
  # or
  uv run python diarizationlm_run.py --input-text "your diarized block ..."

Requires:
  uv add diarizationlm==0.1.5 transformers torch
"""

import argparse
import re
from collections import OrderedDict
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from diarizationlm import utils

MODEL_ID = "google/DiarizationLM-8b-Fisher-v2"
PROMPT_SUFFIX = " --> "
MAX_PROMPT_CHARS = 6000  # includes the suffix, per model card


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-file", type=str, help="Path to diarized text file.")
    src.add_argument("--input-text", type=str, help="Raw diarized text in-line.")
    ap.add_argument("--output-file", type=str, default=None, help="Optional file to save the transferred completion.")
    ap.add_argument("--max-prompt-chars", type=int, default=MAX_PROMPT_CHARS, help="Max chars incl. suffix.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def read_text(args: argparse.Namespace) -> str:
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return f.read()
    return args.input_text


_SPK_PATTERNS = [
    # [SPEAKER_00]: text
    re.compile(r'^\s*\[(?P<label>[^\]\:]+)\]\s*:\s*(?P<text>.*)$'),
    # SPEAKER_00: text
    re.compile(r'^\s*(?P<label>[^\]:]+)\s*:\s*(?P<text>.*)$'),
]


def convert_lines_to_tags(
    diarized_text: str,
) -> Tuple[List[str], Dict[str, int]]:
    """
    Convert input lines with labels like [SPEAKER_00]: to '<speaker:N> ' prefix.
    Preserves line breaks and order. Speaker IDs are assigned in first-appearance order: 1, 2, 3, ...
    Returns:
      converted_lines: list of strings, each beginning with <speaker:N> ...
      label2id: mapping from original labels to integer ids
    """
    label2id: "OrderedDict[str,int]" = OrderedDict()
    next_id = 1
    converted_lines: List[str] = []

    for raw_line in diarized_text.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            # preserve blank lines
            converted_lines.append("")
            continue

        matched = None
        label = None
        text = None
        for pat in _SPK_PATTERNS:
            m = pat.match(line)
            if m:
                matched = m
                label = m.group("label").strip()
                text = m.group("text").strip()
                break

        if matched is None:
            # No label pattern → treat as continuation by the last seen speaker if available,
            # else keep as-is without a speaker tag.
            if label2id:
                last_label = next(reversed(label2id))
                spk_id = label2id[last_label]
                converted_lines.append(f"<speaker:{spk_id}> {line.strip()}")
            else:
                converted_lines.append(line)
            continue

        if label not in label2id:
            label2id[label] = next_id
            next_id += 1
        spk_id = label2id[label]
        converted_lines.append(f"<speaker:{spk_id}> {text}")

    return converted_lines, label2id


def chunk_by_chars_preserving_lines(
    lines: List[str],
    max_total_chars: int,
    suffix: str,
) -> List[str]:
    """
    Pack lines into chunks so that len(chunk) + len(suffix) <= max_total_chars.
    Never split a line across chunks.
    """
    hard_limit = max_total_chars - len(suffix)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0  # length of '\n'.join(cur)

    for line in lines:
        # consider adding this line (plus a newline if cur not empty)
        extra = len(line) + (1 if cur else 0)
        if cur_len + extra > hard_limit:
            # flush current chunk
            chunks.append("\n".join(cur))
            cur = [line]
            cur_len = len(line)
        else:
            if cur:
                cur_len += 1  # newline
            cur.append(line)
            cur_len += len(line)

    if cur:
        chunks.append("\n".join(cur))

    return chunks


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        trust_remote_code=True,
    )
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    # avoid pad warnings
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model


@torch.inference_mode()
def generate_for_chunk(tokenizer, model, hypothesis_chunk: str, verbose: bool = False) -> Tuple[str, str]:
    """
    Returns (completion_text, transferred_text) for one chunk.
    """
    prompt = hypothesis_chunk + PROMPT_SUFFIX
    inputs = tokenizer([prompt], return_tensors="pt")
    # place inputs on the same device as the first param of the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # conservative generation length; model card example uses ~1.2x input tokens
    in_len = int(inputs["input_ids"].shape[1])
    max_new = max(32, min(512, int(in_len * 1.2)))

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # decode only the newly generated tokens
    gen_only = outputs[:, in_len:]
    completion = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]

    transferred = utils.transfer_llm_completion(completion, hypothesis_chunk)

    if verbose:
        print("-----")
        print("PROMPT (tail):", prompt[-200:])
        print("COMPLETION:", completion)
        print("TRANSFERRED (head):", transferred[:200])

    return completion, transferred


def main():
    args = parse_args()
    raw = read_text(args)

    # 1) Convert to <speaker:N> tags while preserving lines
    converted_lines, label2id = convert_lines_to_tags(raw)

    # 2) Chunk by char length so that (chunk + " --> ") <= max chars
    chunks = chunk_by_chars_preserving_lines(
        converted_lines,
        max_total_chars=args.max_prompt_chars,
        suffix=PROMPT_SUFFIX,
    )

    if args.verbose:
        print(f"Speakers mapping: {label2id}")
        print(f"Total chunks: {len(chunks)}")
        for i, c in enumerate(chunks, 1):
            print(f"[Chunk {i}] chars={len(c)} (+{len(PROMPT_SUFFIX)} suffix)")

    # 3) Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # 4) Run generation per chunk and stitch results
    all_transferred: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        if args.verbose:
            print(f"\n=== Processing chunk {idx}/{len(chunks)} ===")
        _, transferred = generate_for_chunk(tokenizer, model, chunk, verbose=args.verbose)
        all_transferred.append(transferred)

    final_text = "\n".join(all_transferred)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(final_text)
    else:
        print("\n" + "=" * 40)
        print("Transferred completion:")
        print("=" * 40)
        print(final_text)


if __name__ == "__main__":
    main()
