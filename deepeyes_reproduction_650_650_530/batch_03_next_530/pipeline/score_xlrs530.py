#!/usr/bin/env python3
"""Deterministically score the second XLRS-Lite sample drawn for GeoEyes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


MULTI_CATEGORY = "Land use classification/Overall Land use classification"
FULL_CATEGORY_COUNTS = {
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
EXPECTED_SAMPLE_COUNTS = {
    category: 10 if full_count == 60 else 50
    for category, full_count in FULL_CATEGORY_COUNTS.items()
}


def extract_answer(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"(?:^|\s)([A-Ea-e])(?:$|[\s,.])", text)
    if not matches and " and " in text.lower():
        matches = re.findall(r"(?:^|\s|and\s+)([A-Ea-e])(?:$|\s|,|\sand)", text.lower())
    if not matches and "," in text:
        matches = re.findall(r"(?:^|\s|,\s*)([A-Ea-e])(?:$|\s|,)", text)
    if not matches:
        matches = re.findall(r"\(([A-Ea-e])\)", text)
    if not matches:
        matches = re.findall(r"[A-Ea-e]", text)
    return "".join(sorted(set(match.upper() for match in matches)))


def matches_answer(predicted: str, answer: str, multi: bool) -> bool:
    predicted_set = set(predicted.upper())
    answer_set = set(answer.upper())
    return predicted_set == answer_set if multi else bool(predicted_set & answer_set)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_ids = [row.get("sample_id") for row in rows]
    if len(rows) != 530:
        raise ValueError(f"expected 530 inference results, found {len(rows)}")
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != 530:
        raise ValueError("results must contain 530 unique, non-empty sample IDs")

    observed_counts = Counter(row.get("category") for row in rows)
    if dict(observed_counts) != EXPECTED_SAMPLE_COUNTS:
        raise ValueError(
            "unexpected category counts: "
            f"expected {EXPECTED_SAMPLE_COUNTS}, observed {dict(observed_counts)}"
        )

    category_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    status_stats: Counter[str] = Counter()
    scored = []
    total_tool_calls = 0
    for row in rows:
        category = row["category"]
        multi = bool(row.get("is_multi")) or category.startswith(MULTI_CATEGORY)
        predicted = str(row.get("extracted_answer") or "")
        correct = matches_answer(predicted, str(row.get("answer") or ""), multi) if predicted else False
        extraction_pass = 1
        if not correct:
            recovered = extract_answer(str(row.get("pred_ans") or ""))
            if recovered and recovered != predicted:
                predicted = recovered
                correct = matches_answer(predicted, str(row.get("answer") or ""), multi)
                extraction_pass = 2

        tool_calls = sum(
            message.get("content", "").count("<tool_call>")
            for message in row.get("pred_output", [])
            if message.get("role") == "assistant" and isinstance(message.get("content"), str)
        )
        total_tool_calls += tool_calls
        status = str(row.get("status") or "unknown")
        status_stats[status] += 1
        category_stats[category]["total"] += 1
        category_stats[category]["correct"] += int(correct)
        scored.append(
            {
                "sample_id": row["sample_id"],
                "category": category,
                "answer": row.get("answer"),
                "predicted": predicted,
                "correct": correct,
                "extraction_pass": extraction_pass,
                "tool_calls": tool_calls,
                "status": status,
            }
        )

    categories = {}
    weighted_correct_equivalent = 0.0
    for category in sorted(FULL_CATEGORY_COUNTS):
        counts = category_stats[category]
        accuracy = counts["correct"] / counts["total"]
        weighted_correct_equivalent += accuracy * FULL_CATEGORY_COUNTS[category]
        categories[category] = {**counts, "accuracy": accuracy, "full_dataset_count": FULL_CATEGORY_COUNTS[category]}

    total_correct = sum(item["correct"] for item in scored)
    summary = {
        "scoring": "GeoEyes/ZEABL deterministic two-pass extraction; zero LLM judge calls",
        "single_choice": "non-empty predicted/ground-truth option-set intersection",
        "multi_choice": "exact predicted/ground-truth option-set equality",
        "total_samples": len(scored),
        "total_correct": total_correct,
        "micro_accuracy_530": total_correct / len(scored),
        "full_dataset_samples": sum(FULL_CATEGORY_COUNTS.values()),
        "weighted_correct_equivalent_3080": weighted_correct_equivalent,
        "weighted_accuracy_3080": weighted_correct_equivalent / sum(FULL_CATEGORY_COUNTS.values()),
        "categories": categories,
        "status_stats": dict(sorted(status_stats.items())),
        "tool_usage": {
            "total_calls": total_tool_calls,
            "average_calls_per_sample": total_tool_calls / len(scored),
            "samples_with_tool": sum(item["tool_calls"] > 0 for item in scored),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scored_path = args.output.with_name(f"{args.output.stem}_scored.jsonl")
    scored_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in scored),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
