#!/usr/bin/env python3
"""Score a 760-sample XLRS result with the pipeline's original rule logic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "pipeline" / "score_xlrs650.py"
SPEC = importlib.util.spec_from_file_location("geoeyes_score", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"unable to load scoring source: {SOURCE}")
SOURCE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE_MODULE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-results", type=int, default=760)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample_ids = [row.get("sample_id") for row in rows]
    if len(rows) != args.expected_results:
        raise ValueError(f"expected {args.expected_results} inference results, found {len(rows)}")
    if None in sample_ids or len(set(sample_ids)) != args.expected_results:
        raise ValueError(f"results must contain {args.expected_results} unique, non-empty sample IDs")

    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    status_stats: dict[str, int] = defaultdict(int)
    tool_calls = 0
    scored = []
    for row in rows:
        category = row.get("category", "unknown")
        multi = bool(row.get("is_multi")) or category.startswith(SOURCE_MODULE.MULTI_CATEGORY)
        predicted = row.get("extracted_answer", "")
        correct = SOURCE_MODULE.is_correct(predicted, row.get("answer", ""), multi) if predicted else False
        if not correct:
            recovered = SOURCE_MODULE.extract_answer(row.get("pred_ans", ""))
            if recovered and recovered != predicted:
                predicted = recovered
                correct = SOURCE_MODULE.is_correct(predicted, row.get("answer", ""), multi)
        calls = sum(
            message.get("content", "").count("<tool_call>")
            for message in row.get("pred_output", [])
            if message.get("role") == "assistant" and isinstance(message.get("content"), str)
        )
        tool_calls += calls
        status_stats[row.get("status", "unknown")] += 1
        category_stats[category]["total"] += 1
        category_stats[category]["correct"] += int(correct)
        scored.append({
            "sample_id": row.get("sample_id"),
            "category": category,
            "answer": row.get("answer"),
            "predicted": predicted,
            "correct": correct,
            "tool_calls": calls,
            "status": row.get("status", "unknown"),
        })

    correct_total = sum(item["correct"] for item in scored)
    categories = {
        category: {
            **counts,
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
        }
        for category, counts in sorted(category_stats.items())
    }
    summary = {
        "scoring": "GeoEyes official rule extraction; independent LLM judge not run",
        "total_samples": len(scored),
        "total_correct": correct_total,
        "overall_accuracy": correct_total / len(scored) if scored else 0.0,
        "categories": categories,
        "status_stats": dict(sorted(status_stats.items())),
        "tool_usage": {
            "total_calls": tool_calls,
            "average_calls_per_sample": tool_calls / len(scored) if scored else 0.0,
            "samples_with_tool": sum(item["tool_calls"] > 0 for item in scored),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_name(f"xlrs{args.expected_results}_scored.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in scored),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
