#!/usr/bin/env python3
"""Verify the two-stage Qwen3-VL WeaveEarth XLRS experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


FULL_ROWS = 3080
FIRST_ROWS = 650
REMAINING_ROWS = FULL_ROWS - FIRST_ROWS
SUPPORT_REGION_COUNT = 6


def parse_args() -> argparse.Namespace:
    root = Path("/root/autodl-tmp/otws/weaveearth/output")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=root / "xlrs-qwen3vl-650" / "results.jsonl")
    parser.add_argument("--remaining", type=Path, default=root / "xlrs-qwen3vl-remaining-2430" / "results.jsonl")
    parser.add_argument("--output", type=Path, default=root / "xlrs-qwen3vl-full" / "results.summary.json")
    return parser.parse_args()


def latest_records(path: Path) -> tuple[dict[int, dict[str, Any]], int, int]:
    latest: dict[int, dict[str, Any]] = {}
    valid = invalid = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            latest[int(record["dataset_position"])] = record
            valid += 1
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            invalid += 1
    return latest, valid, invalid


def manifest_positions(results_path: Path) -> set[int]:
    manifest = json.loads(results_path.with_suffix(".selection.json").read_text(encoding="utf-8"))
    return {int(position) for position in manifest["dataset_positions"]}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_phase(name: str, records: dict[int, dict[str, Any]], expected: set[int], expected_size: int) -> None:
    require(len(expected) == expected_size, f"{name}: manifest has {len(expected)}, expected {expected_size}")
    require(set(records) == expected, f"{name}: record positions do not match manifest")
    bad = [position for position, record in records.items() if record.get("status") != "ok"]
    require(not bad, f"{name}: {len(bad)} latest records are not ok")
    retrieval_bad = [
        position for position, record in records.items()
        if record.get("routing_info", {}).get("retrieval_mode") != "semantic"
    ]
    require(not retrieval_bad, f"{name}: {len(retrieval_bad)} samples did not use semantic retrieval")
    support_bad = [
        position for position, record in records.items()
        if record.get("routing_info", {}).get("n_support_regions") != SUPPORT_REGION_COUNT
    ]
    require(not support_bad, f"{name}: {len(support_bad)} samples do not have {SUPPORT_REGION_COUNT} support regions")


def main() -> None:
    args = parse_args()
    first, first_lines, first_invalid = latest_records(args.first)
    remaining, remaining_lines, remaining_invalid = latest_records(args.remaining)
    first_expected = manifest_positions(args.first)
    remaining_expected = manifest_positions(args.remaining)

    require(first_invalid == 0 and remaining_invalid == 0, "invalid JSONL line(s) found")
    check_phase("first_650", first, first_expected, FIRST_ROWS)
    check_phase("remaining_2430", remaining, remaining_expected, REMAINING_ROWS)
    require(not (first_expected & remaining_expected), "phase selections overlap")
    require(first_expected | remaining_expected == set(range(FULL_ROWS)), "phase selections do not cover 0..3079")

    all_records = {**first, **remaining}
    correct = sum(bool(record.get("correct")) for record in all_records.values())
    parsed = sum(bool(record.get("prediction")) for record in all_records.values())
    category_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in all_records.values():
        category_records[str(record["category"])].append(record)
    categories = {
        category: {
            "samples": len(values),
            "correct": sum(bool(value.get("correct")) for value in values),
            "parsed_predictions": sum(bool(value.get("prediction")) for value in values),
            "accuracy": sum(bool(value.get("correct")) for value in values) / len(values),
        }
        for category, values in sorted(category_records.items())
    }
    summary = {
        "verification": "passed",
        "total_samples": len(all_records),
        "phase_samples": {"first_650": len(first), "remaining_2430": len(remaining)},
        "jsonl_lines": {"first_650": first_lines, "remaining_2430": remaining_lines},
        "coverage": {"positions": "0..3079", "overlap_count": 0, "missing_count": 0},
        "statuses": dict(Counter(record["status"] for record in all_records.values())),
        "semantic_retrieval_samples": len(all_records),
        "support_regions_per_sample": SUPPORT_REGION_COUNT,
        "correct": correct,
        "parsed_predictions": parsed,
        "accuracy": correct / FULL_ROWS,
        "categories": categories,
        "sources": {"first_650": str(args.first), "remaining_2430": str(args.remaining)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
