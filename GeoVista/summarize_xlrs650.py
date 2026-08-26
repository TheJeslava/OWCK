#!/usr/bin/env python3
"""Validate and summarize a GeoVista run against the exact x05 XLRS-650 set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_xlrs650_8k import TASK_PAIRS


IDENTITY_FIELDS = ("category", "index", "path", "question", "answer")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def identity(record: dict) -> tuple:
    return tuple(record[field] for field in IDENTITY_FIELDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("x05/results/zoomearth-x05-balanced-650.jsonl"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("GeoVista/results/x05-balanced-650.local-selection.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = read_jsonl(args.results)
    references = read_jsonl(args.reference_results)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    expected_counts = Counter({category: 50 for category in TASK_PAIRS})
    selected_positions = {
        int(position)
        for entry in selection["categories"].values()
        for position in entry["dataset_positions"]
    }
    references_by_identity = {identity(record): record for record in references}

    checks = {
        "result_lines_650": len(results) == 650,
        "reference_lines_650": len(references) == 650,
        "unique_local_positions_650": len(
            {int(record["dataset_position"]) for record in results}
        )
        == 650,
        "unique_x05_positions_650": len(
            {int(record["x05_dataset_position"]) for record in results}
        )
        == 650,
        "x05_positions_exact_0_to_649": sorted(
            int(record["x05_dataset_position"]) for record in results
        )
        == list(range(650)),
        "identities_match_x05_exactly": {identity(record) for record in results}
        == set(references_by_identity),
        "local_positions_match_selection": {
            int(record["dataset_position"]) for record in results
        }
        == selected_positions,
        "categories_13_by_50": Counter(record["category"] for record in results)
        == expected_counts,
        "all_status_ok": all(record.get("status") == "ok" for record in results),
        "all_context_limit_32768": all(
            record.get("context_limit") == 32768 for record in results
        ),
        "all_x05_position_identity_links_match": all(
            int(references_by_identity[identity(record)]["dataset_position"])
            == int(record["x05_dataset_position"])
            for record in results
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"XLRS-650 validation failed: {failed}")

    category_metrics = {}
    for category in TASK_PAIRS:
        records = [record for record in results if record["category"] == category]
        correct = sum(bool(record.get("correct")) for record in records)
        category_metrics[category] = {
            "samples": len(records),
            "correct": correct,
            "accuracy": correct / len(records),
        }

    context_observations = [
        (length, int(record["dataset_position"]), int(record["x05_dataset_position"]))
        for record in results
        for length in record.get("context_lengths", [])
    ]
    longest_context, longest_local_position, longest_x05_position = max(
        context_observations, default=(0, -1, -1)
    )
    correct = sum(bool(record.get("correct")) for record in results)
    summary = {
        "results": str(args.results),
        "reference_results": str(args.reference_results),
        "selection": str(args.selection),
        "model_path": sorted({record["model_path"] for record in results}),
        "context_limit": 32768,
        "samples": len(results),
        "status_counts": dict(Counter(record["status"] for record in results)),
        "correct": correct,
        "micro_accuracy": correct / len(results),
        "total_inference_seconds": sum(
            float(record.get("total_seconds", 0.0)) for record in results
        ),
        "total_crops": sum(int(record.get("crop_count", 0)) for record in results),
        "turn_distribution": dict(
            sorted(Counter(int(record.get("turns", 0)) for record in results).items())
        ),
        "max_context_tokens": longest_context,
        "max_context_local_position": longest_local_position,
        "max_context_x05_position": longest_x05_position,
        "validation": checks,
        "categories": category_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
