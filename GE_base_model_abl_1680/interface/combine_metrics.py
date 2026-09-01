#!/usr/bin/env python3
"""Combine disjoint XLRS metric files into one summary."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def load_input(value: str) -> tuple[str, Path, dict]:
    if "=" not in value:
        raise ValueError(f"input must use LABEL=PATH syntax: {value}")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path)
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if sum(item["total"] for item in metrics["categories"].values()) != metrics["total_samples"]:
        raise ValueError(f"{path}: category totals do not match total_samples")
    if sum(item["correct"] for item in metrics["categories"].values()) != metrics["total_correct"]:
        raise ValueError(f"{path}: category correct counts do not match total_correct")
    return label, path, metrics


def portable_source(path: Path, output: Path) -> str:
    try:
        return str(path.resolve().relative_to(output.parent.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="LABEL=metrics.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=1680)
    args = parser.parse_args()

    inputs = [load_input(value) for value in args.input]
    category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    status_counts: dict[str, int] = defaultdict(int)
    component_runs = {}
    tool_calls = 0
    samples_with_tool = 0

    for label, path, metrics in inputs:
        component_runs[label] = {
            "source": portable_source(path, args.output),
            "total_samples": metrics["total_samples"],
            "total_correct": metrics["total_correct"],
            "overall_accuracy": metrics["overall_accuracy"],
        }
        for category, counts in metrics["categories"].items():
            category_counts[category]["correct"] += counts["correct"]
            category_counts[category]["total"] += counts["total"]
        for status, count in metrics.get("status_stats", {}).items():
            status_counts[status] += count
        tool_calls += metrics.get("tool_usage", {}).get("total_calls", 0)
        samples_with_tool += metrics.get("tool_usage", {}).get("samples_with_tool", 0)

    categories = {
        category: {
            **counts,
            "accuracy": counts["correct"] / counts["total"],
        }
        for category, counts in sorted(category_counts.items())
    }
    total_samples = sum(item[2]["total_samples"] for item in inputs)
    total_correct = sum(item[2]["total_correct"] for item in inputs)
    if total_samples != args.expected_total:
        raise ValueError(f"expected {args.expected_total} samples, found {total_samples}")
    if total_samples != sum(item["total"] for item in categories.values()):
        raise RuntimeError("combined category totals do not match total_samples")
    if total_correct != sum(item["correct"] for item in categories.values()):
        raise RuntimeError("combined category correct counts do not match total_correct")

    summary = {
        "scoring": inputs[0][2]["scoring"],
        "combination": "three disjoint XLRS selections: 760 + 520 + 400",
        "component_runs": component_runs,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy": total_correct / total_samples,
        "categories": categories,
        "status_stats": dict(sorted(status_counts.items())),
        "tool_usage": {
            "total_calls": tool_calls,
            "average_calls_per_sample": tool_calls / total_samples,
            "samples_with_tool": samples_with_tool,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
