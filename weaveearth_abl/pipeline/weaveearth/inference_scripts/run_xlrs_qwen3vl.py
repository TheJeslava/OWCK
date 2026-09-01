#!/usr/bin/env python3
"""Run the official WeaveEarth inference path on XLRS-Bench-lite.

The adapter materializes XLRS Arrow rows, including its separately stored
multiple-choice options, and embedded image. Region retrieval, answer
extraction, matching, and Qwen calls are delegated to the official
``eval_lrs_vqa_qwen3vl.py`` module.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image as PILImage
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
WEAVE_SOURCE = ROOT / "inference_scripts" / "eval_lrs_vqa_qwen3vl.py"
WEAVE_SOURCE_SHA256 = hashlib.sha256(WEAVE_SOURCE.read_bytes()).hexdigest()
FULL_ROWS = 3080
SAMPLES_PER_CATEGORY = 50
N_CATEGORIES = 13
ADAPTER_PROTOCOL = "xlrs_options_pristine_scoring_v2"
MULTI_SELECT_CATEGORY = "Land use classification/Overall Land use classification"
SMOKE_CATEGORIES = (
    "Complex reasoning/Anomaly Detection and Interpretation",
    "Complex reasoning/Environmental condition reasoning",
    "Complex reasoning/Route planning",
    "Counting/Counting with changing detection",
    "Counting/Counting with complex reasoning",
    "Counting/Overall counting",
    "Land use classification/Overall Land use classification",
    "Land use classification/Regional Land use classification",
    "Object properties/Object classification",
    "Object spatial relationship/Object spatial relationship",
)

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
    offsets = [(2 * p * upper + denominator) // (2 * denominator) for p in range(count)]
    if len(set(offsets)) != count:
        raise RuntimeError(f"selection offsets are not unique: total={total}, count={count}")
    return offsets


def load_dataset(path: Path, split: str) -> Dataset:
    loaded = load_from_disk(str(path))
    if isinstance(loaded, DatasetDict):
        return loaded[split]
    if isinstance(loaded, Dataset):
        return loaded
    raise TypeError(f"expected Dataset or DatasetDict, got {type(loaded).__name__}")


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
        offsets = evenly_spaced_offsets(available, SAMPLES_PER_CATEGORY)
        positions = [rows[offset][1] for offset in offsets]
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


def smoke_positions(dataset: Dataset) -> tuple[list[int], dict]:
    """Select ten deterministic rows covering the distinct XLRS task formats."""
    if len(dataset) != FULL_ROWS:
        raise ValueError(f"smoke-10 selection requires {FULL_ROWS} rows, got {len(dataset)}")
    categories = [str(value) for value in dataset["category"]]
    indices = [int(value) for value in dataset["index"]]
    observed = Counter(categories)
    if observed != Counter(CATEGORY_COUNTS):
        raise ValueError(f"unexpected category counts: {dict(observed)}")

    selected_by_category = {}
    selected = []
    for category in SMOKE_CATEGORIES:
        rows = sorted(
            (sample_index, position)
            for position, (row_category, sample_index) in enumerate(zip(categories, indices))
            if row_category == category
        )
        sample_index, position = rows[len(rows) // 2]
        selected_by_category[category] = {
            "available": len(rows),
            "dataset_position": position,
            "index": sample_index,
        }
        selected.append(position)
    selected = sorted(selected)
    if len(selected) != 10 or len(set(selected)) != 10:
        raise RuntimeError("smoke selection is not exactly 10 unique positions")
    return selected, {
        "dataset_id": "initiacms/XLRS-Bench-lite",
        "dataset_revision": "e540ee2aa745ce9a83784ae76541ddb7f79f03ac",
        "selection": "deterministic midpoint of 10 task-format categories",
        "source_rows": FULL_ROWS,
        "samples": len(selected),
        "categories": selected_by_category,
        "dataset_positions": selected,
    }


def resolve_selection(dataset: Dataset, selection: str) -> tuple[list[int], dict]:
    if selection == "smoke-10":
        return smoke_positions(dataset)
    first, manifest = balanced_positions(dataset)
    if selection == "650":
        return first, manifest
    if selection == "remaining":
        selected_set = set(first)
        positions = [p for p in range(FULL_ROWS) if p not in selected_set]
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
    """Materialize XLRS's separately stored choices as part of its question."""
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
        f"{question}\n\n"
        "The choices are listed below:\n"
        + "\n".join(options)
        + "\n\nXLRS answer format:\n"
        + answer_protocol
    )


def image_bytes_and_path(
    dataset: Dataset, position: int, source_path: str, cache_dir: Path
) -> tuple[bytes, Path, int]:
    payload = dataset._data.column("image")[position].as_py()
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"position {position} has no embedded images")
    image_blobs = [item.get("bytes") if isinstance(item, dict) else None for item in payload]
    if any(not blob for blob in image_blobs):
        raise ValueError(f"position {position} has an image without embedded bytes")

    # infer_one() in the pristine implementation accepts one image path. For
    # XLRS rows with two time points, pass the first source image unchanged;
    # creating a contact sheet would alter the official image preprocessing.
    image_bytes = bytes(image_blobs[0])
    digest = hashlib.sha256(image_bytes).hexdigest()
    suffix = Path(source_path).suffix.lower()
    if not suffix or len(suffix) > 10:
        suffix = ".img"

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}{suffix}"
    if path.exists():
        if path.read_bytes() != image_bytes:
            raise ValueError(f"cached image mismatch: {path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(image_bytes)
        temporary.replace(path)
    return bytes(image_bytes), path, len(image_blobs)


def read_records(path: Path) -> dict[int, dict]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["dataset_position"])] = record
    return records


def assert_resume_protocol(path: Path) -> None:
    """Prevent mixing records scored under different answer protocols."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("adapter_protocol") != ADAPTER_PROTOCOL:
            raise ValueError(
                f"cannot resume {path}: it contains records from a different adapter "
                f"protocol; use a new --output path for {ADAPTER_PROTOCOL}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("/root/autodl-tmp/data/XLRS-Bench-lite-3080"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--selection", choices=("smoke-10", "650", "remaining"), required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "ckpts" / "Qwen3-VL-8B-Instruct")
    parser.add_argument("--semantic-model-path", type=Path, default=ROOT / "ckpts" / "siglip2-so400m-patch16-naflex")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-cache-dir", type=Path, default=ROOT / "output" / "xlrs-image-cache")
    parser.add_argument("--semantic-cache-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun-category",
        help="rerun all selected rows in one category even when their latest record is ok",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--attn-implementation", choices=("flash_attention_2", "sdpa", "eager"), default="sdpa")
    parser.add_argument("--semantic-batch-size", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume: {args.output}")
    PILImage.MAX_IMAGE_PIXELS = None
    dataset = load_dataset(args.data_path, args.split)
    metadata = dataset.remove_columns("image")
    positions, selection_manifest = resolve_selection(dataset, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        assert_resume_protocol(args.output)
    args.output.with_suffix(".selection.json").write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    previous = read_records(args.output) if args.resume else {}
    todo = [
        position
        for position in positions
        if (
            position not in previous
            or previous[position].get("status") != "ok"
            or (args.rerun_category and metadata[position]["category"] == args.rerun_category)
        )
    ]
    print(json.dumps({
        "selection": args.selection,
        "available_rows": len(dataset),
        "selected_rows": len(positions),
        "todo_rows": len(todo),
        "model_path": str(args.model_path),
        "output": str(args.output),
    }, ensure_ascii=False), flush=True)
    if not todo:
        return

    args.save_path = str(args.output.parent)
    args.reasoning_model_path = str(args.model_path)
    args.semantic_model_path = str(args.semantic_model_path)
    args.semantic_cache_dir = str(args.semantic_cache_dir or (args.output.parent / "semantic_cache"))
    args.resize_size = 2048
    args.router_resize_size = 2048
    args.disable_routing = False
    args.routing_grid_rows = 6
    args.routing_grid_cols = 6
    args.routing_overlap_ratio = 0.1
    args.router_max_regions = 8
    args.neighbor_expand = True
    args.use_minimal_support_set = True
    args.use_structured_metadata = True
    args.use_topology_board = True
    args.support_set_budget = 6
    args.support_iou_threshold = 0.4
    args.evidence_patch_size = 448
    args.use_semantic_support = True
    args.semantic_score_weight = 1.0
    args.semantic_anchor_bonus = 0.08
    args.semantic_global_weight = 0.35

    print(f"Loading Qwen3-VL: {args.model_path}", flush=True)
    model, used_attn = WEAVE.load_model_with_fallback(str(args.model_path), args.attn_implementation)
    processor = AutoProcessor.from_pretrained(str(args.model_path), padding_side="left")
    semantic = None
    semantic_status = {"requested": True, "ready": False}
    try:
        semantic = WEAVE.SemanticEncoder(model_path=str(args.semantic_model_path), batch_size=args.semantic_batch_size)
        semantic_status["ready"] = True
    except Exception as exc:
        semantic_status["error"] = str(exc)
        print(f"Semantic encoder unavailable, using router fallback: {exc}", flush=True)

    manifest = {
        "runner": str(Path(__file__).resolve()),
        "weaveearth_source": str(WEAVE_SOURCE),
        "weaveearth_inference_sha256": WEAVE_SOURCE_SHA256,
        "data_path": str(args.data_path),
        "model_path": str(args.model_path),
        "semantic_model_path": str(args.semantic_model_path),
        "selection": args.selection,
        "selected": len(positions),
        "used_attn_implementation": used_attn,
        "max_new_tokens": args.max_new_tokens,
        "adapter_protocol": ADAPTER_PROTOCOL,
        "question_policy": "xlrs_options_and_selection_protocol",
        "multi_image_policy": "first_image_only",
        "semantic_encoder": semantic_status,
        "weaveearth_args": {
            "routing_grid_rows": 6, "routing_grid_cols": 6,
            "routing_overlap_ratio": 0.1, "router_max_regions": 8,
            "neighbor_expand": True, "use_minimal_support_set": True,
            "support_set_budget": 6, "evidence_patch_size": 448,
            "use_topology_board": True, "use_structured_metadata": True,
            "use_semantic_support": True,
        },
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False, "use_cache": True}
    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8", buffering=1) as handle:
        for ordinal, position in enumerate(todo, 1):
            row = metadata[position]
            started = time.monotonic()
            try:
                _, image_path, image_count = image_bytes_and_path(
                    dataset, position, str(row["path"]), args.image_cache_dir
                )
                question = xlrs_question(row)
                answer = str(row["answer"]).strip()
                raw_output, routing_info = WEAVE.infer_one(
                    processor=processor,
                    model=model,
                    semantic_encoder=semantic,
                    image_path=str(image_path),
                    question=question,
                    category=str(row["category"]).strip().lower(),
                    args=args,
                    gen_kwargs=gen_kwargs,
                )
                prediction, extraction_source = WEAVE.extract_model_answer(raw_output)
                correct, prediction_normalized, answer_normalized, match_method = WEAVE.is_match(
                    prediction, answer
                )
                record = {
                    "dataset_position": position,
                    "index": int(row["index"]),
                    "path": row["path"],
                    "category": row["category"],
                    "question": question,
                    "options": row["multi-choice options"],
                    "answer": answer,
                    "prediction": prediction,
                    "prediction_normalized": prediction_normalized,
                    "answer_normalized": answer_normalized,
                    "correct": correct,
                    "match_method": match_method,
                    "raw_output": raw_output,
                    "answer_extraction_source": extraction_source,
                    "image_count": image_count,
                    "multi_image_policy": "first_image_only",
                    "adapter_protocol": ADAPTER_PROTOCOL,
                    "routing_info": routing_info,
                    "status": "ok",
                    "elapsed_seconds": time.monotonic() - started,
                    "weaveearth_inference_sha256": WEAVE_SOURCE_SHA256,
                }
            except Exception as exc:
                record = {
                    "dataset_position": position,
                    "index": int(row["index"]),
                    "path": row["path"],
                    "category": row["category"],
                    "answer": str(row["answer"]).strip(),
                    "multi_image_policy": "first_image_only",
                    "adapter_protocol": ADAPTER_PROTOCOL,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.monotonic() - started,
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"completed {ordinal}/{len(todo)} position={position} status={record['status']} prediction={record.get('prediction')}", flush=True)


if __name__ == "__main__":
    main()
