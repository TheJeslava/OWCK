#!/usr/bin/env python3
"""Probe GeoVista's real multimodal context capacity on one GPU.

Each probe keeps one XLRS image in the prompt, pads the textual part to the
requested total input length, and asks the model for exactly one token.  This
measures actual Transformers inference allocation; it is not a vLLM
``max_model_len`` reservation test.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
import traceback
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor


IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def make_prompt(processor, image, target_input_tokens: int) -> tuple[str, int]:
    """Build a multimodal Qwen prompt whose processor length hits the target."""
    prefix = (
        "<|im_start|>system\nYou are testing long-context multimodal inference."
        "<|im_end|>\n<|im_start|>user\n"
        + IMAGE_TOKEN
        + "\nThe following context is padding for a capacity probe."
    )
    suffix = "\n<|im_end|>\n<|im_start|>assistant\n"

    # The image placeholder expands to many input positions.  Measure that
    # expansion through the multimodal processor, then fill the remaining
    # positions with a one-token textual filler.
    fixed_len = int(
        processor(text=[prefix + suffix], images=[image], return_tensors="pt")["input_ids"].shape[1]
    )
    repetitions = max(0, target_input_tokens - fixed_len)
    prompt = prefix + " x" * repetitions + suffix
    for _ in range(6):
        current = int(
            processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
        )
        delta = target_input_tokens - current
        if delta == 0:
            return prompt, current
        repetitions = max(0, repetitions + delta)
        prompt = prefix + " x" * repetitions + suffix

    # A final exact check is useful for transparent failure instead of silently
    # running a shorter test than requested.
    current = int(
        processor(text=[prompt], images=[image], return_tensors="pt")["input_ids"].shape[1]
    )
    if current != target_input_tokens:
        raise RuntimeError(f"could not construct {target_input_tokens} text tokens (got {current})")
    return prompt, current


def is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("GeoVista-7B-Instruct"))
    parser.add_argument("--data-path", type=Path, default=Path("data/XLRS-Bench-lite-760"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--levels", default="16K,32K,64K,128K")
    parser.add_argument("--output", type=Path, default=Path("GeoVista/results/context-oom.jsonl"))
    args = parser.parse_args()

    levels = []
    for label in args.levels.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*[Kk]\s*", label)
        if not match:
            raise ValueError(f"invalid level {label!r}; use e.g. 16K,32K")
        levels.append((label.strip().upper(), int(match.group(1)) * 1024))

    dataset = load_from_disk(str(args.data_path))["train"]
    original = dataset[args.sample]["image"][0].convert("RGB")
    original.thumbnail((args.image_size, args.image_size))

    print(f"Loading {args.model_path}...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(args.model_path), max_pixels=128 * 128 * 28 * 28
    )
    device = next(model.parameters()).device
    max_positions = int(
        getattr(model.config, "max_position_embeddings", 0)
        or getattr(getattr(model.config, "text_config", None), "max_position_embeddings", 0)
        or 0
    )
    print(
        f"device={device} max_position_embeddings={max_positions} "
        f"attn={getattr(model.config, '_attn_implementation', 'unknown')}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", buffering=1) as handle:
        for label, requested_total in levels:
            # Reserve one position for the generated token.  The 128K model
            # config is 128000 (not 131072), so cap that level accordingly.
            target_total = requested_total
            if max_positions:
                target_total = min(target_total, max_positions)
            target_input = target_total - 1
            started = time.monotonic()
            record: dict[str, object] = {
                "level": label,
                "requested_total_tokens": requested_total,
                "target_total_tokens": target_total,
                "target_input_tokens": target_input,
                "sample": args.sample,
                "image_size": list(original.size),
                "status": "error",
            }
            try:
                prompt, prompt_tokens = make_prompt(processor, original, target_input)
                inputs = processor(
                    text=[prompt], images=[original], return_tensors="pt", padding="longest"
                ).to(device)
                actual_input = int(inputs["input_ids"].shape[1])
                record["actual_input_tokens"] = actual_input
                record["image_grid_thw"] = inputs.get("image_grid_thw").tolist()
                record["image_placeholder_tokens"] = int(
                    (inputs["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
                    .sum()
                )
                if actual_input != target_input:
                    raise RuntimeError(
                        f"processor changed input length: target={target_input}, actual={actual_input}"
                    )

                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        use_cache=True,
                    )
                record["generated_tokens"] = int(generated.shape[1] - actual_input)
                record["generated_id"] = int(generated[0, -1])
                record["status"] = "ok"
                del generated
            except Exception as error:  # continue to the next level after OOM
                record["status"] = "oom" if is_oom(error) else "error"
                record["error"] = f"{type(error).__name__}: {error}"
                record["traceback"] = traceback.format_exc()
            finally:
                record["peak_allocated_mib"] = round(
                    torch.cuda.max_memory_allocated() / 2**20, 1
                )
                record["peak_reserved_mib"] = round(
                    torch.cuda.max_memory_reserved() / 2**20, 1
                )
                record["seconds"] = round(time.monotonic() - started, 2)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps(record, ensure_ascii=False), flush=True)
                if "inputs" in locals():
                    del inputs
                if "prompt" in locals():
                    del prompt
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
