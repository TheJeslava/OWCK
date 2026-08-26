#!/usr/bin/env python3
"""Map the exact x05 XLRS-650 identities onto a local dataset revision."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk

from run_xlrs650_8k import TASK_PAIRS


IDENTITY_FIELDS = ("category", "index", "path", "question", "answer")


def identity(record: dict) -> tuple:
    return tuple(record[field] for field in IDENTITY_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("x05/results/zoomearth-x05-balanced-650.jsonl"),
    )
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/XLRS-Bench-lite-760")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("GeoVista/results/x05-balanced-650.local-selection.json"),
    )
    args = parser.parse_args()

    references = [
        json.loads(line)
        for line in args.reference_results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(references) != 650:
        raise ValueError(f"expected 650 x05 records, got {len(references)}")
    if Counter(record["category"] for record in references) != Counter(
        {category: 50 for category in TASK_PAIRS}
    ):
        raise ValueError("x05 reference is not 13 categories with 50 records each")

    dataset = load_from_disk(str(args.data_path))["train"].remove_columns(["image"])
    local_by_identity: dict[tuple, list[int]] = defaultdict(list)
    for position, record in enumerate(dataset):
        local_by_identity[identity(record)].append(position)

    mapped: list[tuple[dict, int]] = []
    for reference in references:
        matches = local_by_identity[identity(reference)]
        if len(matches) != 1:
            raise ValueError(
                f"x05 position {reference['dataset_position']} has {len(matches)} local matches"
            )
        mapped.append((reference, matches[0]))

    local_positions = [position for _, position in mapped]
    if len(set(local_positions)) != 650:
        raise ValueError("multiple x05 records map to the same local dataset position")

    local_counts = Counter(dataset["category"])
    manifest = {
        "samples": 650,
        "samples_per_category": 50,
        "source_reference_results": str(args.reference_results),
        "local_data_path": str(args.data_path),
        "identity_fields": list(IDENTITY_FIELDS),
        "categories": {},
    }
    for category in TASK_PAIRS:
        category_rows = [
            (reference, position)
            for reference, position in mapped
            if reference["category"] == category
        ]
        manifest["categories"][category] = {
            "available": local_counts[category],
            "selected": len(category_rows),
            "dataset_positions": [position for _, position in category_rows],
            "x05_dataset_positions": [
                int(reference["dataset_position"]) for reference, _ in category_rows
            ],
            "indices": [int(reference["index"]) for reference, _ in category_rows],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}: 650 unique local positions, 13 x 50")


if __name__ == "__main__":
    main()
