#!/usr/bin/env python3
"""Export XLRS-530-next to the JSON/image layout expected by GeoEyes.

Image bytes are copied directly from the Arrow Image feature instead of being
decoded and recompressed, which keeps this temporary evaluation input small
and byte-faithful to the downloaded dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from datasets import load_from_disk


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/root/autodl-tmp/XLRS-530-next"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/geoeyes530_input"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    dataset = load_from_disk(str(args.source))["train"]
    args.output.mkdir(parents=True)
    images_dir = args.output / "images"
    images_dir.mkdir()
    categories = sorted(set(dataset["category"]))
    (args.output / "categories.json").write_text(
        json.dumps({category: [] for category in categories}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Keep raw Arrow chunks so Image bytes are copied without PIL round-trips.
    image_chunks = dataset.data.column("image").chunks
    chunk_rows = []
    for chunk in image_chunks:
        values = chunk.values
        byte_values = values.field("bytes")
        path_values = values.field("path")
        offsets = chunk.offsets.to_pylist()
        rows = []
        for row in range(len(chunk)):
            rows.append([
                (byte_values[image_index].as_py(), path_values[image_index].as_py())
                for image_index in range(offsets[row], offsets[row + 1])
            ])
        chunk_rows.extend(rows)

    if len(chunk_rows) != len(dataset):
        raise RuntimeError(f"image rows {len(chunk_rows)} != dataset rows {len(dataset)}")

    metadata_dataset = dataset.remove_columns("image")
    for row_number, row in enumerate(metadata_dataset):
        uid = f"{safe_name(row['category'])}__{int(row['index']):04d}__{row_number:04d}"
        image_paths: list[str] = []
        for image_number, (payload, source_path) in enumerate(chunk_rows[row_number]):
            if not payload:
                raise ValueError(f"empty image bytes at row {row_number}, image {image_number}")
            suffix = ".jpg" if payload[:2] == b"\xff\xd8" else ".img"
            image_name = f"{uid}__image{image_number}{suffix}"
            (images_dir / image_name).write_bytes(payload)
            image_paths.append(f"images/{image_name}")

        annotation = {
            "question": row["question"],
            "options": row["multi-choice options"],
            "answer": row["answer"],
            "category": row["category"],
            "original_path": row["path"],
            "original_index": int(row["index"]),
            "unique_id": uid,
        }
        if len(image_paths) == 1:
            annotation["image_path"] = image_paths[0]
        else:
            annotation["image_paths"] = image_paths
            annotation["is_multi_image"] = True
        (args.output / f"{uid}.json").write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps({"rows": len(dataset), "categories": len(categories), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
