"""Score XLRS outputs with Text-Before-Vision's deterministic rules."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "zoomearth-xlrs.jsonl"
MULTI_SELECT_CATEGORY = "Land use classification/Overall Land use classification"
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
TBV_PREFIXES = (
    "The answer is", "The best answer is", "The correct answer is",
    "The answers are", "The best answers are", "The correct answers are",
    "The answer", "Answer", "答案", "答案是",
)


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def answer_payload(text: str) -> tuple[str, bool]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1).strip(), True) if match else (text, False)


def extract_tbv_letters(text: str) -> str:
    if not text:
        return ""
    for prefix in TBV_PREFIXES:
        text = text.replace(prefix, "")
    matches = re.findall(r"\(([A-Ea-e])\)", text)
    if not matches:
        matches = re.findall(r"(?:^|\s)([A-Ea-e])(?:$|[\s,.])", text)
    if not matches:
        matches = re.findall(r"[A-Ea-e]", text)
    return "" if not matches else "".join(sorted({match.upper() for match in matches}))


def extract_tbv_letters_improved(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"(?:^|\s)([A-Ea-e])(?:$|[\s,.])", text)
    if matches:
        return "".join(sorted({match.upper() for match in matches}))
    if " and " in text.lower():
        matches = re.findall(r"(?:^|\s|and\s+)([A-Ea-e])(?:$|\s|,|\sand)", text.lower())
        if matches:
            return "".join(sorted({match.upper() for match in matches}))
    if "," in text:
        matches = re.findall(r"(?:^|\s|,\s*)([A-Ea-e])(?:$|\s|,)", text)
        if matches:
            return "".join(sorted({match.upper() for match in matches}))
    matches = re.findall(r"\(([A-Ea-e])\)", text)
    if not matches:
        matches = re.findall(r"[A-Ea-e]", text)
    return "" if not matches else "".join(sorted({match.upper() for match in matches}))


def final_generated_text(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("stage2"):
        return str(record["stage2"]), "stage2"
    if record.get("stage2_output"):
        return str(record["stage2_output"]), "stage2"
    if record.get("stage1"):
        return str(record["stage1"]), "stage1"
    return str(record.get("stage1_output") or ""), "stage1"


def compare_answers(prediction: str, ground_truth: str, is_multi: bool) -> bool:
    predicted, expected = set(prediction.upper()), set(ground_truth.upper())
    return predicted == expected if is_multi else bool(predicted & expected)


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    ground_truth = "".join(sorted(set(str(record.get("ground_truth") or record.get("answer") or "").upper())))
    category = str(record.get("category") or "unknown")
    is_multi = category == MULTI_SELECT_CATEGORY
    raw_text, answer_source = final_generated_text(record)
    payload, wrapped = answer_payload(raw_text)
    first_prediction = extract_tbv_letters(payload)
    prediction, extraction_pass = first_prediction, 1
    correct = compare_answers(prediction, ground_truth, is_multi)
    if not correct:
        improved = extract_tbv_letters_improved(payload)
        if improved and improved != prediction:
            prediction, extraction_pass = improved, 2
            correct = compare_answers(prediction, ground_truth, is_multi)
    has_multi_format = "and" in payload.lower() or "," in payload or len(payload.strip().split()) > 2
    length_similar = bool(prediction) and abs(len(prediction) - len(ground_truth)) <= 1
    llm_candidate = bool(is_multi and not correct and prediction and (has_multi_format or length_similar))
    return {
        "dataset_position": record.get("dataset_position"), "index": record.get("index"),
        "category": category, "ground_truth": ground_truth, "prediction": prediction,
        "correct": correct, "score": int(correct),
        "protocol": "tbv_multi_exact_set" if is_multi else "tbv_single_set_intersection",
        "answer_source": answer_source, "tbv_wrapped_answer": wrapped,
        "tbv_first_prediction": first_prediction, "tbv_extraction_pass": extraction_pass,
        "tbv_llm_judge_candidate": llm_candidate, "tbv_llm_judge_called": False,
        "error": record.get("error"),
    }


def evaluate_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluations = [evaluate_record(record) for record in records]
    categories = defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0})
    protocol_counts = defaultdict(int)
    for item in evaluations:
        values = categories[item["category"]]
        values["correct"] += int(item["correct"]); values["total"] += 1
        values["errors"] += int(bool(item["error"])); protocol_counts[item["protocol"]] += 1
    per_category = {name: {**values, "accuracy": values["correct"] / values["total"]} for name, values in sorted(categories.items())}
    weighted_correct = sum(values["accuracy"] * FULL_CATEGORY_COUNTS.get(name, 0) for name, values in per_category.items())
    full_samples = sum(FULL_CATEGORY_COUNTS.values())
    correct = sum(int(item["correct"]) for item in evaluations)
    return {
        "protocol": {
            "single_choice": "TBV two-pass extraction and non-empty set intersection",
            "multi_choice": "TBV two-pass extraction and exact option-set equality",
            "llm_judge_calls": 0,
            "llm_judge_candidates": sum(int(item["tbv_llm_judge_candidate"]) for item in evaluations),
            "llm_judge_status": "not configured; deterministic TBV rule result retained",
            "counts": dict(sorted(protocol_counts.items())),
        },
        "samples": len(evaluations), "correct": correct,
        "errors": sum(int(bool(item["error"])) for item in evaluations),
        "accuracy": correct / len(evaluations), "full_dataset_samples": full_samples,
        "category_count_weighted_correct_equivalent": weighted_correct,
        "category_count_weighted_accuracy": weighted_correct / full_samples,
        "categories": per_category,
    }, evaluations


def main() -> None:
    parser = argparse.ArgumentParser(description="Score XLRS with TBV answer rules")
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--evaluation-output", type=Path)
    args = parser.parse_args()
    metrics, evaluations = evaluate_records(read_records(args.results_file))
    metrics_path = args.metrics_output or args.results_file.with_suffix(".tbv-metrics.json")
    evaluation_path = args.evaluation_output or args.results_file.with_suffix(".tbv-evaluation.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with evaluation_path.open("w", encoding="utf-8") as output_file:
        for item in evaluations:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote per-sample evaluation to {evaluation_path}")


if __name__ == "__main__":
    main()
