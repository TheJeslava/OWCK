#!/usr/bin/env python3
"""Export the local XLRS-650 DatasetDict in GeoEyes' evaluation format."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

from datasets import DatasetDict, load_from_disk


ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("XLRS_DATASET_SOURCE", ROOT.parent / "data" / "XLRS-650"))
OUTPUT = Path(os.environ.get("XLRS_OUTPUT_DIR", ROOT.parent / "data" / "xlrsbench650"))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"output already exists: {OUTPUT}")
    loaded = load_from_disk(str(SOURCE))
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    images = OUTPUT / "images"
    images.mkdir(parents=True)
    categories = sorted(set(dataset["category"]))
    (OUTPUT / "categories.json").write_text(
        json.dumps({category: [] for category in categories}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for row_number, row in enumerate(dataset):
        uid = f"{safe_name(row['category'])}__{row['index']:04d}__{row_number:04d}"
        image_paths: list[str] = []
        for image_number, image in enumerate(row["image"]):
            image_name = f"{uid}__image{image_number}.jpg"
            image.save(images / image_name, format="JPEG", quality=95)
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
        (OUTPUT / f"{uid}.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"rows": len(dataset), "categories": len(categories)}, indent=2))


if __name__ == "__main__":
    main()
