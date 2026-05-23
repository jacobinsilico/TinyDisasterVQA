#!/usr/bin/env python3
"""
Download only the COCO images required by the pruned COCO-QA dataset.

Input:
  data/processed/required_images.jsonl

Output:
  data/images/train2014/*.jpg
  data/images/val2014/*.jpg
  data/processed/download_report.json
  data/processed/missing_images.jsonl

Features:
  - skips already downloaded images
  - downloads to temporary .part files first
  - retries failed downloads
  - supports --limit for testing
  - supports parallel downloads
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    return items


def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def download_one(
    item: dict,
    min_bytes: int,
    timeout: int,
    retries: int,
    sleep_seconds: float,
) -> dict:
    image_path = Path(item["image_path"])
    image_path.parent.mkdir(parents=True, exist_ok=True)

    url = item["coco_url"]
    part_path = image_path.with_suffix(image_path.suffix + ".part")

    result = {
        "image_id": item["image_id"],
        "coco_split": item["coco_split"],
        "file_name": item["file_name"],
        "image_path": str(image_path),
        "coco_url": url,
        "status": None,
        "error": None,
        "size_bytes": None,
    }

    # Already downloaded.
    if image_path.exists() and image_path.stat().st_size >= min_bytes:
        result["status"] = "exists"
        result["size_bytes"] = image_path.stat().st_size
        return result

    # Remove broken partial file.
    if part_path.exists():
        part_path.unlink()

    last_error: Optional[str] = None

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()

            if len(data) < min_bytes:
                raise RuntimeError(
                    f"Downloaded file too small: {len(data)} bytes"
                )

            with part_path.open("wb") as f:
                f.write(data)

            part_path.replace(image_path)

            result["status"] = "downloaded"
            result["size_bytes"] = image_path.stat().st_size
            return result

        except Exception as e:
            last_error = f"attempt {attempt}/{retries}: {repr(e)}"

            if part_path.exists():
                part_path.unlink()

            if attempt < retries:
                time.sleep(sleep_seconds)

    result["status"] = "failed"
    result["error"] = last_error
    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--required-images",
        type=Path,
        default=Path("data/processed/required_images.jsonl"),
        help="JSONL file produced by 03_resolve_coco_images.py",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/processed/download_report.json"),
    )
    parser.add_argument(
        "--missing-out",
        type=Path,
        default=Path("data/processed/missing_images.jsonl"),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Download timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per image.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep time between retries.",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1024,
        help="Minimum acceptable image file size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Download only first N images for testing. Use 0 for all.",
    )

    args = parser.parse_args()

    print(f"Reading required images: {args.required_images}")
    required_images = read_jsonl(args.required_images)

    if args.limit > 0:
        required_images = required_images[: args.limit]
        print(f"LIMIT MODE: downloading/checking first {args.limit} images")

    print(f"Images to check/download: {len(required_images)}")
    print(f"Workers: {args.num_workers}")

    results = []

    iterator = None

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                download_one,
                item,
                args.min_bytes,
                args.timeout,
                args.retries,
                args.sleep_seconds,
            )
            for item in required_images
        ]

        completed_iter = as_completed(futures)

        if tqdm is not None:
            iterator = tqdm(completed_iter, total=len(futures), desc="Downloading")
        else:
            iterator = completed_iter

        for future in iterator:
            results.append(future.result())

    exists = [r for r in results if r["status"] == "exists"]
    downloaded = [r for r in results if r["status"] == "downloaded"]
    failed = [r for r in results if r["status"] == "failed"]

    report = {
        "num_requested": len(required_images),
        "num_exists": len(exists),
        "num_downloaded": len(downloaded),
        "num_failed": len(failed),
        "failed_preview": failed[:20],
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)

    with args.report_out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_jsonl(failed, args.missing_out)

    print()
    print("Download summary:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print()
    print(f"Report:         {args.report_out}")
    print(f"Missing images: {args.missing_out}")

    if failed:
        raise SystemExit(
            f"Download completed with {len(failed)} failed images. "
            f"Re-run the script to retry; existing files will be skipped."
        )


if __name__ == "__main__":
    main()