#!/usr/bin/env python3
"""Run the unchanged GeoEyes evaluator against an Arrow-backed XLRS split."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from datasets import DatasetDict, load_from_disk
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "pipeline" / "evaluation" / "eval_multi_xlrsbench2.py"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default="None")
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--eval-model-name", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Optional manifest of original Arrow row positions to evaluate.",
    )
    return parser.parse_args()


def load_evaluator(args: argparse.Namespace):
    original_argv = sys.argv
    sys.argv = [
        str(EVALUATOR),
        "--model_name", args.model_name,
        "--api_key", args.api_key,
        "--api_url", args.api_url,
        "--xlrsbench_path", str(args.dataset_path),
        "--save_path", str(args.save_path),
        "--eval_model_name", args.eval_model_name,
        "--num_workers", "1",
    ]
    try:
        spec = importlib.util.spec_from_file_location("geoeyes_evaluator", EVALUATOR)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"unable to load evaluator: {EVALUATOR}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = original_argv


@contextmanager
def materialize_sample(row: dict, sample_position: int, work_dir: Path):
    uid = f"{safe_name(row['category'])}__{row['index']:04d}__{sample_position:04d}"
    with tempfile.TemporaryDirectory(prefix="xlrs760-", dir=work_dir) as directory:
        sample_dir = Path(directory)
        image_dir = sample_dir / "images"
        image_dir.mkdir()
        image_paths: list[str] = []
        for image_number, image in enumerate(row["image"]):
            image_name = f"{uid}__image{image_number}.jpg"
            image.save(image_dir / image_name, format="JPEG", quality=95)
            image_paths.append(f"images/{image_name}")

        annotation = {
            "question": row["question"],
            "options": row["multi-choice options"],
            "answer": row["answer"],
            "category": row["category"],
            "original_path": row["path"],
            "original_index": row["index"],
            "unique_id": uid,
        }
        if len(image_paths) == 1:
            annotation["image_path"] = image_paths[0]
        else:
            annotation["image_paths"] = image_paths
            annotation["is_multi_image"] = True
        annotation_path = sample_dir / f"{uid}.json"
        annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        yield annotation_path


def rebuild_category_file(results_file: Path, category: str, output_file: Path) -> None:
    with results_file.open("r", encoding="utf-8") as source, output_file.open("w", encoding="utf-8") as target:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("category") == category:
                target.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator(args)
    loaded = load_from_disk(str(args.dataset_path))
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    categories_column = dataset.select_columns(["category"])["category"]
    indices_column = dataset.select_columns(["index"])["index"]
    rows_by_category: dict[str, list[tuple[int, int]]] = defaultdict(list)
    if args.selection_manifest:
        manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        records = manifest.get("records", [])
        positions = [record.get("official_position") for record in records]
        if len(positions) != len(set(positions)) or any(
            not isinstance(position, int) or position < 0 or position >= len(dataset)
            for position in positions
        ):
            raise ValueError("selection manifest contains invalid or duplicate positions")
        for record, position in zip(records, positions):
            if record.get("category") != categories_column[position]:
                raise ValueError(f"selection category mismatch at dataset position {position}")
            sample_position = record.get("selected_position", position)
            if not isinstance(sample_position, int) or sample_position < 0:
                raise ValueError(f"invalid sample ID position at dataset position {position}")
            rows_by_category[categories_column[position]].append((position, sample_position))
    else:
        for row_number, category in enumerate(categories_column):
            rows_by_category[category].append((row_number, row_number))
    categories = sorted(rows_by_category)

    output_dir = args.save_path / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "xlrsbench_results.jsonl"
    processed = evaluator.load_processed_sample_ids(str(results_file))
    print(f"processed samples: {len(processed)}")

    with results_file.open("a", encoding="utf-8") as result_stream:
        for category in categories:
            selected_rows = [
                (row_number, sample_position)
                for row_number, sample_position in rows_by_category[category]
                if f"{safe_name(category)}__{indices_column[row_number]:04d}__{sample_position:04d}" not in processed
            ]
            for row_number, sample_position in tqdm(selected_rows, desc=f"Processing XLRSBench {category}"):
                row = dataset[row_number]
                with materialize_sample(row, sample_position, args.work_dir) as annotation_path:
                    result = evaluator.process((str(annotation_path), category))
                result_stream.write(json.dumps(result) + "\n")
                result_stream.flush()
            category_file = output_dir / f"result_{category.replace('/', '_')}_{args.model_name}.jsonl"
            rebuild_category_file(results_file, category, category_file)
            print(f"completed category: {category}")

    print(f"all results written to {results_file}")


if __name__ == "__main__":
    main()
