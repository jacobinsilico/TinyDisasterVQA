#!/usr/bin/env python3
"""
Download and manage COCO images based on pruned COCO-QA splits.
"""

import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_coco_annotation_map(ann_path: Path, coco_split: str) -> dict[int, dict]:
    if not ann_path.exists():
        print(f"Annotation file not found: {ann_path}. Using standard URL scheme resolution.")
        return {}
    
    print(f"Loading annotations: {ann_path}...")
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    img_map = {}
    for img in data.get("images", []):
        img_id = int(img["id"])
        file_name = img["file_name"]
        coco_url = img.get("coco_url", f"http://images.cocodataset.org/{coco_split}/{file_name}")
        img_map[img_id] = {
            "file_name": file_name,
            "coco_split": coco_split,
            "coco_url": coco_url
        }
    return img_map


def resolve_image(image_id: int, coco_map: dict[int, dict]) -> dict:
    if image_id in coco_map:
        return coco_map[image_id]
    
    # Fallback to standard conventions: in COCO-QA, all image_ids map to either val2014 or train2014.
    # We resolve statically by checking val2014 first or train2014.
    # Standard format: COCO_{split}_{image_id:012d}.jpg
    # Since we can check standard locations, we can construct both. Let's return both possible options
    # or default to val2014 (which is very common for COCO-QA) or try to resolve.
    # To be precise, we assume val2014 first unless it's in the annotation map.
    # If the user provides the annotations, it resolves 100% correctly.
    file_name = f"COCO_val2014_{image_id:012d}.jpg"
    coco_split = "val2014"
    coco_url = f"http://images.cocodataset.org/val2014/{file_name}"
    return {
        "file_name": file_name,
        "coco_split": coco_split,
        "coco_url": coco_url
    }


def download_one_image(item: dict, image_root: Path, timeout: int, retries: int) -> dict:
    image_path = image_root / item["coco_split"] / item["file_name"]
    image_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "image_id": item["image_id"],
        "status": "exists",
        "error": None
    }

    if image_path.exists() and image_path.stat().st_size > 1024:
        return result

    part_path = image_path.with_suffix(image_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(item["coco_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
            
            if len(data) < 1024:
                raise RuntimeError("File too small")
            
            with part_path.open("wb") as f:
                f.write(data)
            
            part_path.replace(image_path)
            result["status"] = "downloaded"
            return result
        except Exception as e:
            if part_path.exists():
                part_path.unlink()
            result["error"] = f"attempt {attempt}: {str(e)}"
    
    result["status"] = "failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument(
        "--train-ann",
        type=Path,
        default=Path("data/coco-annotations/train/instances_train2014.json"),
    )
    parser.add_argument(
        "--val-ann",
        type=Path,
        default=Path("data/coco-annotations/val/instances_val2014.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--delete-extra", action="store_true")
    parser.add_argument("--confirm-delete", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)

    args = parser.parse_args()

    # Read pruned manifests
    train_manifest = args.manifest_dir / "cocoqa_train_pruned.jsonl"
    val_manifest = args.manifest_dir / "cocoqa_val_pruned.jsonl"
    test_manifest = args.manifest_dir / "cocoqa_test_pruned.jsonl"

    if not train_manifest.exists() or not val_manifest.exists() or not test_manifest.exists():
        print(f"ERROR: Pruned manifest files are missing from directory: {args.manifest_dir}")
        sys.exit(1)

    print("Reading pruned split manifests...")
    train_samples = read_jsonl(train_manifest)
    val_samples = read_jsonl(val_manifest)
    test_samples = read_jsonl(test_manifest)

    train_req_ids = set(s["image_id"] for s in train_samples)
    val_req_ids = set(s["image_id"] for s in val_samples)
    test_req_ids = set(s["image_id"] for s in test_samples)
    all_required_ids = train_req_ids | val_req_ids | test_req_ids

    # Load annotations
    train_ann_map = load_coco_annotation_map(args.train_ann, "train2014")
    val_ann_map = load_coco_annotation_map(args.val_ann, "val2014")
    
    global_ann_map = {}
    global_ann_map.update(train_ann_map)
    global_ann_map.update(val_ann_map)

    # Build the required resolved images map
    required_resolved = {}
    for img_id in all_required_ids:
        meta = resolve_image(img_id, global_ann_map)
        # Verify and correct coco_split if we can identify it from train/val annotations
        if img_id in train_ann_map:
            meta = train_ann_map[img_id]
        elif img_id in val_ann_map:
            meta = val_ann_map[img_id]
        
        required_resolved[img_id] = {
            "image_id": img_id,
            "file_name": meta["file_name"],
            "coco_split": meta["coco_split"],
            "coco_url": meta["coco_url"]
        }

    # Find present local images under train2014 and val2014 subdirs
    local_images = {}
    for split in ["train2014", "val2014"]:
        split_dir = args.image_root / split
        if split_dir.exists():
            for f in split_dir.glob("*.jpg"):
                local_images[f.name] = f

    # Calculate status lists
    required_files = {r["file_name"]: r for r in required_resolved.values()}
    required_present = []
    required_missing = []
    
    for r_file, r_item in required_files.items():
        if r_file in local_images:
            required_present.append(r_item)
        else:
            required_missing.append(r_item)

    # Extra local files: present but not required by any split
    extra_local_files = []
    for f_name, f_path in local_images.items():
        if f_name not in required_files:
            extra_local_files.append((f_name, f_path))

    # Report counts
    print("\n==================================================")
    print("IMAGE DATASET MANAGEMENT REPORT")
    print("==================================================")
    print(f"Number of required train split images: {len(train_req_ids)}")
    print(f"Number of required val split images:   {len(val_req_ids)}")
    print(f"Number of required test split images:  {len(test_req_ids)}")
    print(f"Total unique required images:          {len(all_required_ids)}")
    print(f"Number of currently present local:     {len(local_images)}")
    print(f"Number of missing required images:     {len(required_missing)}")
    print(f"Number of extra local images not needed: {len(extra_local_files)}")
    print("==================================================\n")

    # Safety checks for --delete-extra
    if args.delete_extra:
        if not args.confirm_delete:
            print("ERROR: --delete-extra requires explicit confirmation via --confirm-delete.")
            sys.exit(1)
        
        # Verify that we never delete outside image_root/train2014 and image_root/val2014
        for name, path in extra_local_files:
            real_path = path.resolve()
            train_dir = (args.image_root / "train2014").resolve()
            val_dir = (args.image_root / "val2014").resolve()
            is_inside = real_path.is_relative_to(train_dir) or real_path.is_relative_to(val_dir)
            if not is_inside:
                print(f"SAFETY ERROR: Extra image path {path} is outside the allowed directories!")
                sys.exit(1)

    # Dry-run reporting first 20 images
    if args.dry_run:
        print("Dry-run mode active. No files will be downloaded or deleted.")
        if extra_local_files:
            print("\nPreview of extra files that would be deleted (up to 20):")
            for idx, (name, path) in enumerate(sorted(extra_local_files)[:20], 1):
                print(f"  {idx}. {path}")
        if required_missing:
            print("\nPreview of missing files that would be downloaded (up to 20):")
            for idx, item in enumerate(required_missing[:20], 1):
                print(f"  {idx}. ID={item['image_id']} split={item['coco_split']} URL={item['coco_url']}")
        print()
        sys.exit(0)

    # Delete Extra Images
    if args.delete_extra:
        if not extra_local_files:
            print("No extra local images found to delete.")
        else:
            print(f"Deleting {len(extra_local_files)} extra images...")
            deleted_cnt = 0
            for name, path in extra_local_files:
                try:
                    path.unlink()
                    deleted_cnt += 1
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")
            print(f"Successfully deleted {deleted_cnt} extra images.")
            print()

    # Download Missing Images
    if args.download_missing:
        if not required_missing:
            print("No missing images to download.")
        else:
            print(f"Downloading {len(required_missing)} missing images using {args.num_workers} threads...")
            results = []
            
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = [
                    executor.submit(
                        download_one_image,
                        item,
                        args.image_root,
                        args.timeout,
                        args.retries
                    )
                    for item in required_missing
                ]
                
                completed_iter = as_completed(futures)
                if tqdm is not None:
                    iterator = tqdm(completed_iter, total=len(futures), desc="Downloading images")
                else:
                    iterator = completed_iter

                for future in iterator:
                    results.append(future.result())
            
            downloaded = [r for r in results if r["status"] == "downloaded"]
            failed = [r for r in results if r["status"] == "failed"]
            print(f"\nDownload finished: successfully downloaded {len(downloaded)}, failed {len(failed)}")
            if failed:
                print("Failed downloads preview:")
                for idx, f_item in enumerate(failed[:10], 1):
                    print(f"  {idx}. ID={f_item['image_id']} Error={f_item['error']}")
                print()


if __name__ == "__main__":
    main()
