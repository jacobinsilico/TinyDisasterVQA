#!/usr/bin/env python3
"""
Inspect COCO-QA pruned dataset samples visually by saving padded resized images with clear text overlays.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw


# We import AspectRatioPreservingResizeAndPad from src.data.transforms
# but we can also write a lightweight clone here or add repo path to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.transforms import AspectRatioPreservingResizeAndPad


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


def resolve_image_path(image_id: int, image_root: Path) -> Path:
    # Look for COCO image in train2014 and val2014
    for split in ["train2014", "val2014"]:
        f_name = f"COCO_{split}_{image_id:012d}.jpg"
        p = image_root / split / f_name
        if p.exists():
            return p
    return Path()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--answer", type=str, default=None, help="Optional ground-truth answer class filter")
    parser.add_argument("--type", type=str, default=None, choices=["object", "color"], help="Optional type filter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path("debug_samples"))
    parser.add_argument("--image-root", type=Path, default=Path("data/images"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--image-size", type=int, default=128, help="Resize image to this size with letterbox padding")
    parser.add_argument("--padding-fill-rgb", type=str, default="123,116,103", help="RGB padding fill color")

    args = parser.parse_args()

    # Read manifest
    manifest_path = args.manifest_dir / f"cocoqa_{args.split}_pruned.jsonl"
    vocab_path = args.manifest_dir / "answer_vocab.json"
    
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Loading {args.split} manifest: {manifest_path}...")
    samples = read_jsonl(manifest_path)

    # Filter
    if args.type:
        samples = [s for s in samples if s["type"] == args.type]
    if args.answer:
        samples = [s for s in samples if s["answer"].lower() == args.answer.lower()]

    if not samples:
        print("No samples matched the filters.")
        sys.exit(0)

    # Load vocab to verify answer_id if needed
    vocab = {}
    if vocab_path.exists():
        with vocab_path.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
    answer_to_id = vocab.get("answer_to_id", {})

    print(f"Total matched samples: {len(samples)}")
    print(f"Sampling {min(args.num_samples, len(samples))} examples using seed {args.seed}...")

    random.seed(args.seed)
    selected = random.sample(samples, min(args.num_samples, len(samples)))

    args.save_dir.mkdir(parents=True, exist_ok=True)
    fill_rgb = tuple(int(c) for c in args.padding_fill_rgb.split(","))

    # Initialize padded resize transform
    transform = AspectRatioPreservingResizeAndPad(args.image_size, fill_rgb=fill_rgb)

    for idx, sample in enumerate(selected, 1):
        image_id = int(sample["image_id"])
        image_path = resolve_image_path(image_id, args.image_root)

        if not image_path.exists() or image_path == Path():
            print(f"WARNING: Image not found locally for ID={image_id}. Skipping visualization.")
            continue

        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        img_transformed = transform(img)

        # Let's create an expanded image (adding a black bar at the top or bottom for high-contrast legibility)
        # We add 120 pixels of padding at the bottom of the image for text overlays
        canvas_h = img_transformed.height + 140
        canvas = Image.new("RGB", (img_transformed.width, canvas_h), (20, 20, 20))
        canvas.paste(img_transformed, (0, 0))

        # Draw overlays
        draw = ImageDraw.Draw(canvas)
        
        q = sample["question"]
        ans = sample["answer"]
        qtype = sample["type"]
        ans_id = sample.get("answer_id", answer_to_id.get(ans, "N/A"))

        # We wrap long questions into multiple lines if needed
        words = q.split()
        lines = []
        cur_line = []
        for w in words:
            if len(" ".join(cur_line + [w])) > 24:
                lines.append(" ".join(cur_line))
                cur_line = [w]
            else:
                cur_line.append(w)
        if cur_line:
            lines.append(" ".join(cur_line))
        
        wrapped_q = "\n".join(lines)

        text_y = img_transformed.height + 10
        draw.text((10, text_y), f"Q: {wrapped_q}", fill=(255, 255, 255))
        
        text_y_details = canvas_h - 60
        draw.text((10, text_y_details), f"A: {ans} (ID: {ans_id})", fill=(0, 255, 100))
        draw.text((10, text_y_details + 20), f"Type: {qtype} | ID: {image_id}", fill=(180, 180, 180))
        draw.text((10, text_y_details + 40), f"Split: {args.split}", fill=(180, 180, 180))

        # Save to save-dir
        out_name = f"sample_{idx:02d}_id_{image_id}_ans_{ans}.jpg"
        out_path = args.save_dir / out_name
        canvas.save(out_path)
        print(f"Saved: {out_path}")

    print("\nVisual inspection finished. Visualized files are saved in:", args.save_dir)


if __name__ == "__main__":
    main()
