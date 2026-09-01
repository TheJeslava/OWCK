#!/usr/bin/env python3
"""Download the next non-overlapping XLRS-Lite sample round.

The public dataset is a 38 GiB collection of Arrow stream shards.  This
script uses HTTP ranges to fetch only the image buffers for the selected rows,
so the complete 3,080-row dataset never has to be stored locally.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import threading
import time
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image
import pyarrow as pa
import requests


DATASET_ID = "initiacms/XLRS-Bench-lite"
DATASET_REVISION = "e540ee2aa745ce9a83784ae76541ddb7f79f03ac"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
SHARD_COUNT = 74
METADATA_RANGE_END = 131_071
IMAGE_OUTER_OFFSETS_BUFFER = 23
IMAGE_BYTES_OFFSETS_BUFFER = 26
IMAGE_BYTES_DATA_BUFFER = 27
EXPECTED_BUFFER_COUNT = 31


@dataclass(frozen=True)
class ImageSegment:
    row: int
    image: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class ShardPlan:
    index: int
    url: str
    file_size: int
    body_start: int
    body_length: int
    row_start: int
    row_count: int
    local_rows: list[int]
    prefix: bytes
    tail_start: int
    segments: list[ImageSegment]


_thread_state = threading.local()


def session() -> requests.Session:
    current = getattr(_thread_state, "session", None)
    if current is None:
        current = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        current.mount("https://", adapter)
        _thread_state.session = current
    return current


def fetch_range(url: str, start: int, end: int, attempts: int = 5) -> tuple[bytes, int]:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(15, 600),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"server ignored Range header: HTTP {response.status_code}")
            data = response.content
            if len(data) != expected:
                raise RuntimeError(f"short range response: expected {expected}, got {len(data)}")
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or (int(match.group(1)), int(match.group(2))) != (start, end):
                raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
            return data, int(match.group(3))
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed range {start}-{end} from {url}: {last_error}")


def field_offset(buffer: bytes, table: int, field: int) -> int:
    vtable = table - struct.unpack_from("<i", buffer, table)[0]
    vtable_length = struct.unpack_from("<H", buffer, vtable)[0]
    location = vtable + 4 + 2 * field
    if location + 2 > vtable + vtable_length:
        return 0
    return struct.unpack_from("<H", buffer, location)[0]


def indirect(buffer: bytes, location: int) -> int:
    return location + struct.unpack_from("<I", buffer, location)[0]


def struct_vector(buffer: bytes, table: int, field: int, size: int) -> list[int]:
    offset = field_offset(buffer, table, field)
    if not offset:
        return []
    vector = indirect(buffer, table + offset)
    length = struct.unpack_from("<I", buffer, vector)[0]
    return [vector + 4 + size * index for index in range(length)]


def parse_record_batch(raw: bytes) -> tuple[int, int, int, list[tuple[int, int]]]:
    if struct.unpack_from("<I", raw, 0)[0] != 0xFFFFFFFF:
        raise ValueError("Arrow stream schema continuation marker is missing")
    schema_length = struct.unpack_from("<I", raw, 4)[0]
    record_message_start = 8 + schema_length
    if struct.unpack_from("<I", raw, record_message_start)[0] != 0xFFFFFFFF:
        raise ValueError("Arrow record-batch continuation marker is missing")
    metadata_length = struct.unpack_from("<I", raw, record_message_start + 4)[0]
    metadata_start = record_message_start + 8
    metadata = raw[metadata_start : metadata_start + metadata_length]
    body_start = metadata_start + metadata_length

    message = struct.unpack_from("<I", metadata, 0)[0]
    body_length_location = field_offset(metadata, message, 3)
    header_location = field_offset(metadata, message, 2)
    if not body_length_location or not header_location:
        raise ValueError("Arrow message does not contain a record batch")
    body_length = struct.unpack_from("<q", metadata, message + body_length_location)[0]
    record_batch = indirect(metadata, message + header_location)
    row_count = struct.unpack_from(
        "<q", metadata, record_batch + field_offset(metadata, record_batch, 0)
    )[0]
    buffer_locations = struct_vector(metadata, record_batch, 2, 16)
    buffers = [struct.unpack_from("<qq", metadata, location) for location in buffer_locations]
    if len(buffers) != EXPECTED_BUFFER_COUNT:
        raise ValueError(f"unexpected Arrow buffer count: {len(buffers)}")
    return body_start, body_length, row_count, buffers


def build_selection(manifest: dict[str, Any]) -> tuple[set[int], dict[str, dict[str, Any]]]:
    positions: set[int] = set()
    categories: dict[str, dict[str, Any]] = {}
    for category, values in manifest["categories"].items():
        available = int(values["available"])
        selected = set(map(int, values["indices"]))
        category_start = min(map(int, values["dataset_positions"]))
        if available <= 100:
            indices = [index for index in range(available) if index not in selected]
            mode = "all remaining samples"
        else:
            indices = []
            used = set(selected)
            for index in sorted(selected):
                candidate = (index + 1) % available
                while candidate in used:
                    candidate = (candidate + 1) % available
                indices.append(candidate)
                used.add(candidate)
            mode = "next unselected sample after each prior sample, with wraparound"
        dataset_positions = [category_start + index for index in indices]
        if positions.intersection(dataset_positions):
            raise RuntimeError(f"duplicate source position while selecting {category}")
        positions.update(dataset_positions)
        categories[category] = {
            "available": available,
            "previously_selected": len(selected),
            "new_selected": len(indices),
            "selection_mode": mode,
            "dataset_positions": dataset_positions,
            "indices": indices,
        }
    return positions, categories


def plan_shard(
    index: int,
    url: str,
    selected_positions: set[int],
    row_start: int,
) -> ShardPlan:
    raw, file_size = fetch_range(url, 0, METADATA_RANGE_END)
    body_start, body_length, row_count, buffers = parse_record_batch(raw)
    local_rows = [
        position - row_start
        for position in sorted(selected_positions)
        if row_start <= position < row_start + row_count
    ]
    outer_offset, outer_length = buffers[IMAGE_OUTER_OFFSETS_BUFFER]
    byte_offset, byte_offset_length = buffers[IMAGE_BYTES_OFFSETS_BUFFER]
    data_offset, data_length = buffers[IMAGE_BYTES_DATA_BUFFER]
    prefix_end = body_start + data_offset
    if prefix_end > len(raw):
        raise ValueError(f"shard {index}: metadata range is too short")
    outer_offsets = struct.unpack_from(
        f"<{outer_length // 4}i", raw, body_start + outer_offset
    )
    byte_offsets = struct.unpack_from(
        f"<{byte_offset_length // 4}i", raw, body_start + byte_offset
    )
    if len(outer_offsets) != row_count + 1:
        raise ValueError(f"shard {index}: invalid image-list offsets")

    segments = []
    for row in local_rows:
        for image_index in range(outer_offsets[row], outer_offsets[row + 1]):
            start = body_start + data_offset + byte_offsets[image_index]
            end = body_start + data_offset + byte_offsets[image_index + 1] - 1
            if end < start:
                raise ValueError(f"shard {index}, row {row}: empty image bytes")
            segments.append(ImageSegment(row, image_index, start, end))
    tail_start = body_start + data_offset + data_length
    if body_start + body_length + 8 != file_size:
        raise ValueError(f"shard {index}: Arrow stream size mismatch")
    return ShardPlan(
        index=index,
        url=url,
        file_size=file_size,
        body_start=body_start,
        body_length=body_length,
        row_start=row_start,
        row_count=row_count,
        local_rows=local_rows,
        prefix=raw[:prefix_end],
        tail_start=tail_start,
        segments=segments,
    )


def download_segment(plan: ShardPlan, segment: ImageSegment) -> tuple[tuple[int, int], bytes]:
    data, _ = fetch_range(plan.url, segment.start, segment.end)
    return (plan.index, segment.image), data


def materialize_batch(
    plan: ShardPlan,
    downloaded: dict[tuple[int, int], bytes],
) -> pa.RecordBatch:
    tail, _ = fetch_range(plan.url, plan.tail_start, plan.file_size - 1)
    with tempfile.NamedTemporaryFile(prefix=f"xlrs-next-{plan.index:05d}-", suffix=".arrow") as handle:
        handle.truncate(plan.file_size)
        handle.seek(0)
        handle.write(plan.prefix)
        handle.seek(plan.tail_start)
        handle.write(tail)
        for segment in plan.segments:
            data = downloaded[(plan.index, segment.image)]
            if len(data) != segment.length:
                raise RuntimeError(f"shard {plan.index}: downloaded image length changed")
            handle.seek(segment.start)
            handle.write(data)
        handle.flush()

        source = pa.memory_map(handle.name, "r")
        try:
            reader = pa.ipc.open_stream(source)
            batch = reader.read_next_batch()
            if batch.num_rows != plan.row_count:
                raise RuntimeError(f"shard {plan.index}: decoded row count changed")
            selected = batch.take(pa.array(plan.local_rows, type=pa.int64()))
            if selected.num_rows != len(plan.local_rows):
                raise RuntimeError(f"shard {plan.index}: failed to take selected rows")
            return selected
        finally:
            source.close()


def validate_images(table: pa.Table) -> int:
    # Keep source batches separate: combining all image bytes would exceed
    # Arrow's 32-bit list-offset limit once the round is larger than 2 GiB.
    verified = 0
    for chunk in table.column("image").chunks:
        values = chunk.values
        byte_values = values.field("bytes")
        offsets = chunk.offsets.to_pylist()
        for row in range(len(chunk)):
            for image_index in range(offsets[row], offsets[row + 1]):
                payload = byte_values[image_index].as_py()
                if not payload:
                    raise ValueError(f"row {row}, image {image_index}: empty image")
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                verified += 1
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=Path, default=Path("/root/autodl-tmp/XLRS-650"))
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/XLRS-530-next"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/xlrs530-next-cache"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    selection_path = args.existing / "selection_manifest.json"
    prior_manifest = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_positions, categories = build_selection(prior_manifest)
    print(f"Planned {len(selected_positions)} new rows across {len(categories)} categories", flush=True)

    base_url = (
        f"{args.endpoint.rstrip('/')}/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}"
        "/train/data-%05d-of-00074.arrow"
    )
    # This fixed revision has 46 shards of 42 rows and 28 shards of 41 rows.
    # Fetch their metadata concurrently, then verify the complete contiguous layout.
    row_starts = [
        index * 42 if index < 46 else 46 * 42 + (index - 46) * 41
        for index in range(SHARD_COUNT)
    ]
    plans = []
    with ThreadPoolExecutor(max_workers=min(args.workers, 16)) as executor:
        futures = {
            executor.submit(
                plan_shard,
                index,
                base_url % index,
                selected_positions,
                row_starts[index],
            ): index
            for index in range(SHARD_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            plans.append(future.result())
            if completed % 10 == 0 or completed == SHARD_COUNT:
                print(f"Planned source shards: {completed}/{SHARD_COUNT}", flush=True)
    plans.sort(key=lambda plan: plan.index)
    expected_start = 0
    for plan in plans:
        if plan.row_start != expected_start:
            raise RuntimeError(f"shard {plan.index}: non-contiguous row start")
        expected_start += plan.row_count
    if expected_start != 3_080:
        raise RuntimeError(f"source shard rows total {expected_start}, expected 3080")
    planned_rows = sum(len(plan.local_rows) for plan in plans)
    segments = [(plan, segment) for plan in plans for segment in plan.segments]
    selected_bytes = sum(segment.length for _, segment in segments)
    if planned_rows != len(selected_positions):
        raise RuntimeError(f"planned {planned_rows} rows, expected {len(selected_positions)}")
    print(
        f"Downloading {len(segments)} images ({selected_bytes / 2**30:.3f} GiB) for {planned_rows} rows",
        flush=True,
    )

    free_bytes = shutil.disk_usage(args.output.parent).free
    if free_bytes < selected_bytes + 512 * 2**20:
        raise OSError(
            f"insufficient output space: need at least {(selected_bytes + 512 * 2**20) / 2**30:.2f} GiB, "
            f"have {free_bytes / 2**30:.2f} GiB"
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[tuple[int, int], bytes] = {}
    pending_segments = []
    cached_bytes = 0
    for plan, segment in segments:
        cache_path = args.cache_dir / f"shard-{plan.index:05d}-image-{segment.image:05d}.bin"
        if cache_path.is_file() and cache_path.stat().st_size == segment.length:
            downloaded[(plan.index, segment.image)] = cache_path.read_bytes()
            cached_bytes += segment.length
        else:
            pending_segments.append((plan, segment))
    if cached_bytes:
        print(
            f"Reusing {len(downloaded)} cached images ({cached_bytes / 2**30:.3f} GiB)",
            flush=True,
        )
    completed_bytes = cached_bytes
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_segment, plan, segment)
            for plan, segment in pending_segments
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            key, data = future.result()
            downloaded[key] = data
            plan_index, image_index = key
            cache_path = args.cache_dir / f"shard-{plan_index:05d}-image-{image_index:05d}.bin"
            partial_path = cache_path.with_suffix(".part")
            partial_path.write_bytes(data)
            os.replace(partial_path, cache_path)
            completed_bytes += len(data)
            if completed % 20 == 0 or completed == len(futures):
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"Images: {len(downloaded)}/{len(segments)}, "
                    f"{completed_bytes / 2**30:.3f}/{selected_bytes / 2**30:.3f} GiB, "
                    f"{completed_bytes / 2**20 / elapsed:.1f} MiB/s",
                    flush=True,
                )

    batches = []
    for completed, plan in enumerate(plans, 1):
        batches.append(materialize_batch(plan, downloaded))
        for segment in plan.segments:
            downloaded.pop((plan.index, segment.image))
        if completed % 10 == 0 or completed == len(plans):
            print(f"Materialized source shards: {completed}/{len(plans)}", flush=True)
    if downloaded:
        raise RuntimeError(f"unused image downloads remain: {len(downloaded)}")

    table = pa.Table.from_batches(batches)
    if table.num_rows != len(selected_positions):
        raise RuntimeError(f"assembled {table.num_rows} rows, expected {len(selected_positions)}")
    verified_images = validate_images(table)
    if verified_images != len(segments):
        raise RuntimeError(f"verified {verified_images} images, expected {len(segments)}")

    dataset = Dataset(table)
    actual_pairs = list(zip(dataset["category"], map(int, dataset["index"]), strict=True))
    expected_pairs = {
        (category, index)
        for category, values in categories.items()
        for index in values["indices"]
    }
    if set(actual_pairs) != expected_pairs or len(actual_pairs) != len(expected_pairs):
        raise RuntimeError("downloaded category/index pairs do not match the selection plan")
    prior = load_from_disk(str(args.existing))["train"].remove_columns("image")
    prior_pairs = set(zip(prior["category"], map(int, prior["index"]), strict=True))
    if prior_pairs.intersection(expected_pairs):
        raise RuntimeError("new sample set overlaps the previous XLRS-650 set")

    DatasetDict({"train": dataset}).save_to_disk(str(args.output), max_shard_size="1GB")
    manifest = {
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "previous_selection": str(selection_path),
        "sampling": (
            "for categories with <=100 samples select all unselected rows; otherwise select "
            "the next unselected sorted index after each prior row, wrapping to index 0"
        ),
        "samples": len(dataset),
        "images": verified_images,
        "category_count": len(categories),
        "categories": categories,
    }
    (args.output / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    reloaded = load_from_disk(str(args.output))["train"]
    counts = Counter(reloaded["category"])
    expected_counts = Counter(
        {category: values["new_selected"] for category, values in categories.items()}
    )
    if len(reloaded) != len(selected_positions) or counts != expected_counts:
        raise RuntimeError(f"saved dataset validation failed: rows={len(reloaded)}, counts={counts}")
    shutil.rmtree(args.cache_dir)
    print(json.dumps({
        "output": str(args.output),
        "rows": len(reloaded),
        "images": verified_images,
        "size_bytes": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()),
        "categories": dict(sorted(counts.items())),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
