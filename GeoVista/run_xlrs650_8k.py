#!/usr/bin/env python3
"""Run GeoVista locally on the x05 XLRS-650 selection with an 8K context cap.

The released GeoVista evaluator is API-oriented.  This adapter preserves its
SOP/tool-call protocol while using the local Transformers checkpoint directly,
so it can run on a single GPU without a separate vLLM service.
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
from datasets import Image as DatasetImage
from datasets import load_from_disk
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


TASK_PAIRS = [
    "Complex reasoning/Anomaly Detection and Interpretation",
    "Complex reasoning/Environmental condition reasoning",
    "Complex reasoning/Route planning",
    "Counting/Counting with changing detection",
    "Counting/Counting with complex reasoning",
    "Counting/Overall counting",
    "Counting/Regional counting",
    "Land use classification/Overall Land use classification",
    "Land use classification/Regional Land use classification",
    "Object properties/Object classification",
    "Object properties/Object color",
    "Object properties/Object motion state",
    "Object spatial relationship/Object spatial relationship",
]

SYSTEM_PROMPTS = {
    "Complex reasoning/Anomaly Detection and Interpretation": """Task: Complex Visual Reasoning and Interpretation.
Algorithm SOP (STRICT CHECKLIST PROCESS):
1. PLAN PHASE: In your first <think> block, carefully analyze the question, the options, and the global visual context. If fine-grained details or specific regional features are needed to deduce the correct option, plan to inspect that area.
2. EXECUTION RULE: If the target region or contextual clues are unclear in the global view, request a high-resolution crop via:
   <tool_call>{\"name\": \"zoom_in\", \"arguments\": {\"bbox\": [x1, y1, x2, y2]}}</tool_call>
   (Coordinates are 0-1000 relative to the image size).
   STOP GENERATING AFTER THE TOOL CALL.
3. DISCOVERY FORMAT: Upon receiving the cropped image, strictly continue your reasoning inside a new <think> block, evaluating how the new visual evidence aligns with the multiple-choice options.
4. FINAL ANSWER: Once you have confidently deduced the best interpretation, output ONLY the letter of the correct choice wrapped in <answer>...</answer>.""",
    "Complex reasoning/Environmental condition reasoning": """Task: Complex Visual Reasoning and Interpretation.
Algorithm SOP (STRICT CHECKLIST PROCESS):
1. PLAN PHASE: In your first <think> block, carefully analyze the question, the options, and the global visual context. If fine-grained details or specific regional features are needed to deduce the correct option, plan to inspect that area.
2. EXECUTION RULE: If the target region or contextual clues are unclear in the global view, request a high-resolution crop via:
   <tool_call>{\"name\": \"zoom_in\", \"arguments\": {\"bbox\": [x1, y1, x2, y2]}}</tool_call>
   (Coordinates are 0-1000 relative to the image size).
   STOP GENERATING AFTER THE TOOL CALL.
3. DISCOVERY FORMAT: Upon receiving the cropped image, strictly continue your reasoning inside a new <think> block, evaluating how the new visual evidence aligns with the multiple-choice options.
4. FINAL ANSWER: Once you have confidently deduced the best interpretation, output ONLY the letter of the correct choice wrapped in <answer>...</answer>.""",
    "Complex reasoning/Route planning": """Task: Complex Visual Reasoning and Interpretation.
Algorithm SOP (STRICT CHECKLIST PROCESS):
1. PLAN PHASE: In your first <think> block, carefully analyze the question, the options, and the global visual context. If fine-grained details or specific regional features are needed to deduce the correct option, plan to inspect that area.
2. EXECUTION RULE: If the target region or contextual clues are unclear in the global view, request a high-resolution crop via:
   <tool_call>{\"name\": \"zoom_in\", \"arguments\": {\"bbox\": [x1, y1, x2, y2]}}</tool_call>
   (Coordinates are 0-1000 relative to the image size).
   STOP GENERATING AFTER THE TOOL CALL.
3. DISCOVERY FORMAT: Upon receiving the cropped image, strictly continue your reasoning inside a new <think> block, evaluating how the new visual evidence aligns with the multiple-choice options.
4. FINAL ANSWER: Once you have confidently deduced the best interpretation, output ONLY the letter of the correct choice wrapped in <answer>...</answer>.""",
}
for _category in TASK_PAIRS[3:]:
    SYSTEM_PROMPTS.setdefault(_category, """Task: Complex Visual Reasoning and Interpretation.
Algorithm SOP (STRICT CHECKLIST PROCESS):
1. PLAN PHASE: In your first <think> block, analyze the question, choices, and global visual context. If details are unclear, plan a crop.
2. EXECUTION RULE: Request a crop with <tool_call>{\"name\": \"zoom_in\", \"arguments\": {\"bbox\": [x1, y1, x2, y2]}}</tool_call>, using 0-1000 image-relative coordinates, then stop.
3. DISCOVERY FORMAT: After a crop, continue reasoning in a new <think> block.
4. FINAL ANSWER: Output only the choice letter in <answer>...</answer>.""")

SYSTEM_PREFIX = "<|im_start|>system\n"
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"
IM_END = "<|im_end|>\n"


def render_messages(messages: list[dict]) -> tuple[str, list[Image.Image]]:
    """Render the Qwen chat format and return images in placeholder order."""
    parts: list[str] = []
    images: list[Image.Image] = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n")
        if message.get("image") is not None:
            parts.append(IMAGE_TOKEN)
            images.append(message["image"])
        parts.append(message.get("text", ""))
        parts.append(IM_END)
    parts.append("<|im_start|>assistant\n")
    return "".join(parts), images


def resize_max(image: Image.Image, max_size: int) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def encode_bbox_crop(image: Image.Image, bbox: list[float]) -> Image.Image:
    width, height = image.size
    # Keep the crop origin inside the last valid pixel.  Model-generated
    # coordinates can reach or exceed 1000; allowing x1 == width (or y1 ==
    # height) produces a zero-sized PIL image that the Qwen processor cannot
    # resize.
    x1 = max(0, min(width - 1, int(bbox[0] / 1000 * width)))
    y1 = max(0, min(height - 1, int(bbox[1] / 1000 * height)))
    x2 = max(1, min(width, int(bbox[2] / 1000 * width)))
    y2 = max(1, min(height, int(bbox[3] / 1000 * height)))
    if x2 <= x1:
        x2 = min(width, x1 + 16)
    if y2 <= y1:
        y2 = min(height, y1 + 16)
    return image.crop((x1, y1, x2, y2))


def extract_tool(text: str) -> tuple[str, list[float]] | None:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not match:
        match = re.search(r"(\{\s*\"name\"\s*:\s*\"zoom_in\".*?\})", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
        bbox = payload.get("arguments", {}).get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            return None
        values = [float(value) for value in bbox[:4]]
        if values[2] <= values[0] or values[3] <= values[1]:
            return None
        return "zoom_in", values
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def extract_answer(text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    bare = text.strip()
    return bare if re.fullmatch(r"[A-E]+", bare, re.IGNORECASE) else None


def prediction(answer: str | None, text: str) -> str:
    payload = (answer if answer is not None else text).upper()
    letters = re.findall(r"[A-E]", payload)
    return "".join(sorted(set(letters))) if letters else ""


def selection_positions(selection_path: Path) -> tuple[list[int], dict[int, int]]:
    manifest = json.loads(selection_path.read_text())
    categories = manifest.get("categories", {})
    if set(categories) != set(TASK_PAIRS):
        raise ValueError("selection categories do not match the 13 XLRS task pairs")

    positions: list[int] = []
    reference_positions: dict[int, int] = {}
    for category in TASK_PAIRS:
        entry = categories[category]
        category_positions = [int(value) for value in entry["dataset_positions"]]
        if len(category_positions) != 50 or len(set(category_positions)) != 50:
            raise ValueError(f"selection category {category!r} is not 50 unique samples")
        positions.extend(category_positions)

        source_positions = entry.get("x05_dataset_positions", [])
        if source_positions:
            if len(source_positions) != len(category_positions):
                raise ValueError(f"selection category {category!r} has mismatched x05 positions")
            reference_positions.update(
                zip(category_positions, (int(value) for value in source_positions))
            )

    positions.sort()
    if len(positions) != 650 or len(set(positions)) != 650:
        raise ValueError("selection is not 650 unique local dataset positions")
    return positions, reference_positions


def trim_context(messages: list[dict], processor: object, limit: int) -> tuple[str, list[Image.Image], int]:
    """Drop oldest tool exchanges until input leaves room for 1024 new tokens."""
    while True:
        prompt, images = render_messages(messages)
        inputs = processor(text=[prompt], images=images, return_tensors="pt", padding="longest")
        length = int(inputs["input_ids"].shape[1])
        if length + 1024 <= limit or len(messages) <= 3:
            return prompt, images, length
        # Keep system + initial global user, remove the oldest assistant/tool pair.
        del messages[2:4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("GeoVista-7B-Instruct"))
    parser.add_argument("--data-path", type=Path, default=Path("data/XLRS-Bench-lite-760"))
    parser.add_argument("--selection-path", type=Path, default=Path("x05/results/zoomearth-x05-balanced-650.selection.json"))
    parser.add_argument("--output", type=Path, default=Path("GeoVista/results/geovista-instruct-xlrs650-8k.jsonl"))
    parser.add_argument("--context-limit", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    positions, reference_positions = selection_positions(args.selection_path)
    dataset = load_from_disk(str(args.data_path))["train"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[int, dict] = {}
    if args.resume and args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                previous[int(record["dataset_position"])] = record

    print(f"Loading {args.model_path} with context limit {args.context_limit}...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        str(args.model_path), max_pixels=128 * 128 * 28 * 28
    )
    processor.tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    mode = "a" if args.resume and args.output.exists() else "w"

    with args.output.open(mode, encoding="utf-8", buffering=1) as handle:
        for completed, position in enumerate(positions, 1):
            if args.resume and position in previous and previous[position].get("status") == "ok":
                print(f"[{completed}/650] skip {position}", flush=True)
                continue
            started = time.monotonic()
            doc = dataset[position]
            try:
                original = doc["image"][0].convert("RGB")
                global_image = resize_max(original, 1024)
                question = (
                    f"Question: {doc['question']}\n\n[System Observation]\n"
                    "Current View: Global View\nPlease strictly follow the SOP. "
                    "Begin your response with <think>.\n\nThe choices are listed below:\n"
                    + "\n".join(doc["multi-choice options"])
                )
                messages = [
                    {"role": "system", "text": SYSTEM_PROMPTS[doc["category"]]},
                    {"role": "user", "text": question, "image": global_image},
                ]
                outputs: list[str] = []
                context_lengths: list[int] = []
                crop_count = 0
                for turn in range(1, args.max_turns + 1):
                    prompt, images, input_length = trim_context(
                        messages, processor, args.context_limit
                    )
                    context_lengths.append(input_length)
                    inputs = processor(
                        text=[prompt], images=images, return_tensors="pt", padding="longest"
                    ).to(device)
                    max_new = max(1, min(1024, args.context_limit - input_length))
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs, max_new_tokens=max_new, do_sample=True,
                            temperature=0.01, top_p=None, top_k=None,
                        )
                    output = processor.tokenizer.decode(
                        generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                    ).strip()
                    outputs.append(output)
                    messages.append({"role": "assistant", "text": output})
                    answer = extract_answer(output)
                    if answer is not None:
                        break
                    tool = extract_tool(output)
                    if tool is None:
                        break
                    _, bbox = tool
                    crop = resize_max(encode_bbox_crop(original, bbox), 1024)
                    crop_count += 1
                    messages.append({
                        "role": "user",
                        "text": f"System Return: Cropped image for bbox {bbox}. Please continue your analysis INSIDE a new <think> block.",
                        "image": crop,
                    })
                final_output = outputs[-1] if outputs else ""
                final_answer = extract_answer(final_output)
                record = {
                    "dataset_position": position,
                    "x05_dataset_position": reference_positions.get(position),
                    "index": doc["index"],
                    "path": doc["path"],
                    "question": doc["question"],
                    "multi_choice_options": doc["multi-choice options"],
                    "answer": doc["answer"],
                    "category": doc["category"],
                    "model_path": str(args.model_path),
                    "selection_path": str(args.selection_path),
                    "context_limit": args.context_limit,
                    "context_lengths": context_lengths,
                    "turns": len(outputs),
                    "crop_count": crop_count,
                    "status": "ok",
                    "outputs": outputs,
                    "final_answer": final_answer,
                    "prediction": prediction(final_answer, final_output),
                    "correct": prediction(final_answer, final_output) == str(doc["answer"]),
                    "total_seconds": time.monotonic() - started,
                }
            except Exception as error:
                record = {
                    "dataset_position": position,
                    "x05_dataset_position": reference_positions.get(position),
                    "index": doc.get("index"),
                    "category": doc.get("category"),
                    "model_path": str(args.model_path),
                    "selection_path": str(args.selection_path),
                    "context_limit": args.context_limit,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "prediction": "",
                    "correct": False,
                    "total_seconds": time.monotonic() - started,
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{completed}/650] position={position} status={record['status']} "
                f"pred={record.get('prediction','')} gold={doc.get('answer','')} "
                f"turns={record.get('turns',0)} time={record['total_seconds']:.1f}s",
                flush=True,
            )
            del doc
            gc.collect()


if __name__ == "__main__":
    main()
