#!/usr/bin/env python3
"""Run a Qwen3-VL global-thumbnail baseline on selected XLRS-Bench-lite rows.

This intentionally bypasses WeaveEarth evidence construction: it sends only
the source image resized with the same ``resize_long_side(..., 2048)`` policy
used for WeaveEarth's final-stage global thumbnail. The text prompt contains
only the XLRS question, its stored choices, and the XLRS answer protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image as PILImage
from transformers import AutoProcessor


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
WEAVE_ROOT = PIPELINE_ROOT.parent / "weaveearth"
WEAVE_SOURCE = WEAVE_ROOT / "inference_scripts" / "eval_lrs_vqa_qwen3vl.py"
WEAVE_SOURCE_SHA256 = hashlib.sha256(WEAVE_SOURCE.read_bytes()).hexdigest()
ADAPTER_PROTOCOL = "xlrs_qwen3vl_weaveearth_finalstage_text_options_v3"
MULTI_SELECT_CATEGORY = "Land use classification/Overall Land use classification"
FULL_ROWS = 3080
SAMPLES_PER_CATEGORY = 50

CATEGORY_COUNTS = {
    "Complex reasoning/Anomaly Detection and Interpretation": 100,
    "Complex reasoning/Environmental condition reasoning": 100,
    "Complex reasoning/Route planning": 100,
    "Counting/Counting with changing detection": 60,
    "Counting/Counting with complex reasoning": 100,
    "Counting/Overall counting": 60,
    "Counting/Regional counting": 100,
    "Land use classification/Overall Land use classification": 100,
    "Land use classification/Regional Land use classification": 200,
    "Object properties/Object classification": 800,
    "Object properties/Object color": 800,
    "Object properties/Object motion state": 60,
    "Object spatial relationship/Object spatial relationship": 500,
}


def load_weave_module():
    spec = importlib.util.spec_from_file_location("weaveearth_qwen3vl_eval", WEAVE_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not import {WEAVE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WEAVE = load_weave_module()


def load_dataset(path: Path, split: str) -> Dataset:
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        return loaded[split]
    if isinstance(loaded, Dataset):
        return loaded
    raise TypeError(f"expected Dataset or DatasetDict, got {type(loaded).__name__}")


def evenly_spaced_offsets(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0 or count > total:
        raise ValueError(f"invalid selection total={total}, count={count}")
    if count == total:
        return list(range(total))
    if count == 1:
        return [0]
    if count == 2:
        return [0, total - 1]
    upper = total - 2
    denominator = count - 1
    offsets = [(2 * position * upper + denominator) // (2 * denominator) for position in range(count)]
    if len(set(offsets)) != count:
        raise RuntimeError(f"selection offsets are not unique: total={total}, count={count}")
    return offsets


def balanced_positions(dataset: Dataset) -> tuple[list[int], dict]:
    if len(dataset) != FULL_ROWS:
        raise ValueError(f"650/remaining selection requires {FULL_ROWS} rows, got {len(dataset)}")
    categories = [str(value) for value in dataset["category"]]
    indices = [int(value) for value in dataset["index"]]
    observed = Counter(categories)
    if observed != Counter(CATEGORY_COUNTS):
        raise ValueError(f"unexpected category counts: {dict(observed)}")
    selected_by_category = {}
    selected = []
    for category, available in CATEGORY_COUNTS.items():
        rows = sorted(
            (sample_index, position)
            for position, (row_category, sample_index) in enumerate(zip(categories, indices))
            if row_category == category
        )
        positions = [rows[offset][1] for offset in evenly_spaced_offsets(available, SAMPLES_PER_CATEGORY)]
        selected_by_category[category] = {
            "available": available,
            "selected": len(positions),
            "dataset_positions": positions,
            "indices": [int(dataset["index"][position]) for position in positions],
        }
        selected.extend(positions)
    selected = sorted(selected)
    if len(selected) != 650 or len(set(selected)) != 650:
        raise RuntimeError("selection is not exactly 650 unique positions")
    return selected, {
        "dataset_id": "initiacms/XLRS-Bench-lite",
        "dataset_revision": "e540ee2aa745ce9a83784ae76541ddb7f79f03ac",
        "selection": "index-sorted evenly-spaced 50 per category",
        "source_rows": FULL_ROWS,
        "samples": 650,
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "categories": selected_by_category,
        "dataset_positions": selected,
    }


def resolve_selection(dataset: Dataset, selection: str) -> tuple[list[int], dict]:
    first, manifest = balanced_positions(dataset)
    if selection == "650":
        return first, manifest
    if selection == "remaining":
        first_set = set(first)
        positions = [position for position in range(FULL_ROWS) if position not in first_set]
        return positions, {
            "dataset_id": manifest["dataset_id"],
            "dataset_revision": manifest["dataset_revision"],
            "selection": "complement of deterministic 650 selection",
            "source_rows": FULL_ROWS,
            "samples": len(positions),
            "excluded_samples": len(first),
            "dataset_positions": positions,
        }
    raise ValueError(f"unknown selection {selection!r}")


def xlrs_question(row: dict) -> str:
    """Build the WeaveEarth final-stage text structure with XLRS-specific fields."""
    question = str(row["question"]).strip()
    options = [str(option) for option in row["multi-choice options"]]
    labels = "".join(chr(ord("A") + index) for index in range(len(options)))
    if str(row["category"]) == MULTI_SELECT_CATEGORY:
        answer_protocol = (
            "Select every option that applies. Inside <answer>, output only the "
            f"applicable uppercase letters from {labels}, in alphabetical order, "
            "with no spaces or separators."
        )
    else:
        answer_protocol = (
            "Select exactly one best option. Inside <answer>, output exactly one "
            f"uppercase letter from {labels}, with no other text."
        )
    return (
        "Answer the question about the remote sensing image.\n"
        f"Question: {question}\n"
        "The choices are listed below:\n"
        + "\n".join(options)
        + "\nXLRS answer format:\n"
        + answer_protocol
        + "\nReturn only a short answer phrase in this format: <answer>...</answer>"
    )


def load_first_image(dataset: Dataset, position: int) -> tuple[PILImage.Image, int]:
    payload = dataset._data.column("image")[position].as_py()
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"position {position} has no embedded images")
    first = payload[0]
    image_bytes = first.get("bytes") if isinstance(first, dict) else None
    if not image_bytes:
        raise ValueError(f"position {position} has no first image bytes")
    with PILImage.open(BytesIO(bytes(image_bytes))) as source:
        return source.convert("RGB"), len(payload)


def read_completed(output: Path) -> dict[int, dict]:
    records = {}
    if not output.exists():
        return records
    for line in output.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["dataset_position"])] = record
    return records


def assert_resume_protocol(output: Path) -> None:
    for record in read_completed(output).values():
        if record.get("adapter_protocol") != ADAPTER_PROTOCOL:
            raise ValueError(
                f"{output} contains a different adapter protocol; choose a new output path"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("/root/autodl-tmp/data/XLRS-Bench-lite-3080"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--selection", choices=("650", "remaining"), required=True)
    parser.add_argument("--model-path", type=Path, default=WEAVE_ROOT / "ckpts" / "Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output already exists; pass --resume: {args.output}")
    if args.resume:
        assert_resume_protocol(args.output)

    PILImage.MAX_IMAGE_PIXELS = None
    dataset = load_dataset(args.data_path, args.split)
    positions, selection_manifest = resolve_selection(dataset, args.selection)
    metadata = dataset.remove_columns("image")
    previous = read_completed(args.output) if args.resume else {}
    todo = [position for position in positions if previous.get(position, {}).get("status") != "ok"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runner": str(Path(__file__).resolve()),
        "adapter_protocol": ADAPTER_PROTOCOL,
        "dataset_id": "initiacms/XLRS-Bench-lite",
        "selection": args.selection,
        "selected": len(positions),
        "model_path": str(args.model_path),
        "weaveearth_source": str(WEAVE_SOURCE),
        "weaveearth_inference_sha256": WEAVE_SOURCE_SHA256,
        "visual_input": {
            "type": "global_thumbnail_only",
            "source_image_policy": "first_image_only",
            "resize_policy": "resize_long_side",
            "target_long_side": 2048,
        },
        "text_input": "weaveearth_finalstage_text_plus_xlrs_choices_and_answer_protocol",
        "excluded_components": [
            "SigLIP retrieval",
            "routing image and router prompt",
            "grid regions",
            "neighbor expansion",
            "minimal support evidence set",
            "evidence board",
            "structured region metadata",
        ],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".selection.json").write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "selection": args.selection,
        "selected_rows": len(positions),
        "completed_rows": len(positions) - len(todo),
        "todo_rows": len(todo),
        "output": str(args.output),
    }, ensure_ascii=False), flush=True)
    if not todo:
        return
    print(f"Loading Qwen3-VL: {args.model_path}", flush=True)
    model, used_attn = WEAVE.load_model_with_fallback(str(args.model_path), args.attn_implementation)
    processor = AutoProcessor.from_pretrained(str(args.model_path), padding_side="left")
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8", buffering=1) as handle:
        for ordinal, position in enumerate(todo, 1):
            row = metadata[position]
            started = time.monotonic()
            try:
                image, image_count = load_first_image(dataset, position)
                global_thumbnail = WEAVE.resize_long_side(image, 2048)
                prompt = xlrs_question(row)
                raw_output = WEAVE.call_qwen(
                    processor=processor,
                    model=model,
                    images=[global_thumbnail],
                    prompt=prompt,
                    gen_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False, "use_cache": True},
                )
                prediction, extraction_source = WEAVE.extract_model_answer(raw_output)
                correct, prediction_normalized, answer_normalized, match_method = WEAVE.is_match(
                    prediction, str(row["answer"]).strip()
                )
                record = {
                    "dataset_position": position,
                    "index": int(row["index"]),
                    "path": row["path"],
                    "category": row["category"],
                    "question": prompt,
                    "options": row["multi-choice options"],
                    "answer": str(row["answer"]).strip(),
                    "prediction": prediction,
                    "prediction_normalized": prediction_normalized,
                    "answer_normalized": answer_normalized,
                    "correct": correct,
                    "match_method": match_method,
                    "raw_output": raw_output,
                    "answer_extraction_source": extraction_source,
                    "visual_input": {
                        **manifest["visual_input"],
                        "source_size": list(image.size),
                        "input_size": list(global_thumbnail.size),
                        "image_count_in_row": image_count,
                    },
                    "adapter_protocol": ADAPTER_PROTOCOL,
                    "used_attn_implementation": used_attn,
                    "weaveearth_inference_sha256": WEAVE_SOURCE_SHA256,
                    "status": "ok",
                    "elapsed_seconds": time.monotonic() - started,
                }
            except Exception as exc:
                record = {
                    "dataset_position": position,
                    "index": int(row["index"]),
                    "category": row["category"],
                    "adapter_protocol": ADAPTER_PROTOCOL,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.monotonic() - started,
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if ordinal % 10 == 0 or ordinal == len(todo):
                print(
                    f"completed {ordinal}/{len(todo)} position={position} "
                    f"status={record['status']} prediction={record.get('prediction')}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
