#!/usr/bin/env python3
"""Score local GeoEyes outputs with ZEABL's deterministic TBV protocol.

The evaluator itself is imported from the cloned ZEABL repository.  Local
GeoEyes records are adapted only at the field boundary: ``pred_ans`` becomes
ZEABL's stage1 output, because GeoEyes stores the final answer there.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


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
REFERENCE_CATEGORIES = {
    "Object properties/Object classification",
    "Object properties/Object color",
    "Object properties/Object motion state",
}


def load_zeabl_module(path: Path):
    spec = importlib.util.spec_from_file_location("zeabl_eval_xlrs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ZEABL evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def validate_full(rows: list[dict[str, Any]], label: str) -> None:
    ids = [row.get("sample_id") for row in rows]
    if len(rows) != 650 or any(not sample_id for sample_id in ids) or len(set(ids)) != 650:
        raise ValueError(f"{label}: expected 650 unique sample IDs, found {len(rows)}")
    categories = {row.get("category") for row in rows}
    if categories != set(FULL_CATEGORY_COUNTS):
        raise ValueError(f"{label}: category set differs: {sorted(categories)}")
    counts = {category: sum(row.get("category") == category for row in rows) for category in categories}
    if set(counts.values()) != {50}:
        raise ValueError(f"{label}: expected 50 rows per category, got {counts}")


def overlay_reference(base: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_full(base, "base")
    ids = [row.get("sample_id") for row in reference]
    if len(reference) != 150 or any(not sample_id for sample_id in ids) or len(set(ids)) != 150:
        raise ValueError("reference overlay must contain 150 unique sample IDs")
    if {row.get("category") for row in reference} != REFERENCE_CATEGORIES:
        raise ValueError("reference overlay is not exactly the three bbox categories")
    if any(sum(row.get("category") == category for row in reference) != 50 for category in REFERENCE_CATEGORIES):
        raise ValueError("reference overlay must contain 50 rows per bbox category")

    base_by_id = {row["sample_id"]: row for row in base}
    unknown = sorted(set(ids) - set(base_by_id))
    if unknown:
        raise ValueError(f"reference overlay has IDs absent from base: {unknown[:3]}")
    overlay_by_id = {row["sample_id"]: row for row in reference}
    return [overlay_by_id.get(row["sample_id"], row) for row in base]


def adapt_for_zeabl(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adapted = []
    for position, row in enumerate(rows):
        # GeoEyes has already removed the <answer> wrapper into pred_ans.  Put
        # it in stage1_output so ZEABL's final_generated_text() sees it.
        item = dict(row)
        item["dataset_position"] = position
        item["stage1_output"] = str(row.get("pred_ans") or "")
        item["stage2_output"] = ""
        item["error"] = row.get("error") or ("status=error" if row.get("status") == "error" else None)
        adapted.append(item)
    return adapted


def score(label: str, rows: list[dict[str, Any]], evaluator: Any) -> dict[str, Any]:
    validate_full(rows, label)
    metrics, evaluations = evaluator.evaluate_records(adapt_for_zeabl(rows))
    metrics = dict(metrics)
    metrics["label"] = label
    metrics["scoring_source"] = "ZEABL eval_xlrs.py deterministic TBV; no LLM judge"
    metrics["input_rows"] = len(rows)
    metrics["weighted_accuracy_3080"] = metrics["category_count_weighted_accuracy"]
    metrics["weighted_correct_equivalent_3080"] = metrics["category_count_weighted_correct_equivalent"]
    metrics["micro_accuracy_650"] = metrics["accuracy"]
    metrics["evaluations"] = evaluations
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zeabl-evaluator", type=Path, default=Path("/tmp/ZEABL/v0-9/src/eval/eval_xlrs.py"))
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--geo1-reference-150", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evaluator = load_zeabl_module(args.zeabl_evaluator)
    original = read_jsonl(args.original)
    reference = read_jsonl(args.geo1_reference_150)
    geo1 = overlay_reference(original, reference)
    result = {
        "protocol": "ZEABL deterministic TBV: two-pass extraction; single non-empty intersection; multi exact set; zero LLM calls",
        "original": score("original_geoeyes", original, evaluator),
        "geo1_overlay": score("geo1_650_with_bbox150_overlay", geo1, evaluator),
        "provenance": {
            "original": str(args.original),
            "geo1_reference_150": str(args.geo1_reference_150),
            "overlay": "replace base rows by matching sample_id",
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep the console result compact; full per-sample evaluations are in --output.
    print(json.dumps({
        "protocol": result["protocol"],
        "original": {k: result["original"][k] for k in ("correct", "accuracy", "weighted_correct_equivalent_3080", "weighted_accuracy_3080")},
        "geo1_overlay": {k: result["geo1_overlay"][k] for k in ("correct", "accuracy", "weighted_correct_equivalent_3080", "weighted_accuracy_3080")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
