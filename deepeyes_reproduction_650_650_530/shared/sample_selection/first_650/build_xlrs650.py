#!/usr/bin/env python3
"""Build and validate the deterministic 50-per-category XLRS subset."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "XLRS-Bench-lite"
OUTPUT = ROOT / "XLRS-650"
SAMPLES_PER_CATEGORY = 50


def evenly_spaced_offsets(total: int, count: int) -> list[int]:
    if count <= 0 or count > total:
        raise ValueError(f"cannot select {count} samples from {total}")
    if count == total:
        return list(range(total))
    if count == 1:
        return [0]

    # Match the established XLRS balanced-sampling rule: distribute picks from
    # the first through penultimate sorted indices, avoiding endpoint bias.
    upper = total - 2
    denominator = count - 1
    offsets = [
        (2 * position * upper + denominator) // (2 * denominator)
        for position in range(count)
    ]
    if len(set(offsets)) != count:
        raise RuntimeError(f"sampling produced duplicate offsets for {total=}")
    return offsets


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"output already exists: {OUTPUT}")

    loaded = load_from_disk(str(SOURCE))
    dataset: Dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded

    rows_by_category: dict[str, list[tuple[int, int]]] = {}
    for position, (category, index) in enumerate(
        zip(dataset["category"], dataset["index"], strict=True)
    ):
        rows_by_category.setdefault(category, []).append((index, position))

    if len(rows_by_category) != 13:
        raise ValueError(f"expected 13 categories, found {len(rows_by_category)}")

    selected_positions: list[int] = []
    category_manifest: dict[str, dict[str, object]] = {}
    for category, indexed_rows in sorted(rows_by_category.items()):
        indexed_rows.sort()
        offsets = evenly_spaced_offsets(len(indexed_rows), SAMPLES_PER_CATEGORY)
        selected = [indexed_rows[offset] for offset in offsets]
        selected_positions.extend(position for _, position in selected)
        category_manifest[category] = {
            "available": len(indexed_rows),
            "selected": len(selected),
            "dataset_positions": [position for _, position in selected],
            "indices": [index for index, _ in selected],
        }

    selected_positions.sort()
    subset = dataset.select(selected_positions)
    DatasetDict({"train": subset}).save_to_disk(str(OUTPUT), max_shard_size="1GB")

    reloaded = load_from_disk(str(OUTPUT))["train"]
    counts = Counter(reloaded["category"])
    if len(reloaded) != 650 or set(counts.values()) != {50}:
        raise RuntimeError(f"invalid saved subset: rows={len(reloaded)}, counts={counts}")

    manifest = {
        "source": str(SOURCE),
        "sampling": "sort each category by index, then evenly sample first through penultimate",
        "samples": len(reloaded),
        "category_count": len(counts),
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "categories": category_manifest,
    }
    (OUTPUT / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(reloaded), "categories": dict(sorted(counts.items()))}, indent=2))


if __name__ == "__main__":
    main()
