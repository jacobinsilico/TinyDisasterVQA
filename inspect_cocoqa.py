import json
from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict

IMAGE_ROOTS = [
    Path("data/images/train2014"),
    Path("data/images/val2014"),
]

def find_image(image_id):
    candidates = [
        f"COCO_train2014_{int(image_id):012d}.jpg",
        f"COCO_val2014_{int(image_id):012d}.jpg",
        f"{image_id}.jpg",
    ]
    for root in IMAGE_ROOTS:
        for name in candidates:
            p = root / name
            if p.exists():
                return p
    return None

for split in ["train", "val", "test"]:
    manifest = Path(f"data/processed/cocoqa_{split}_pruned.jsonl")
    dims = Counter()
    missing = 0
    seen = set()

    with manifest.open() as f:
        for line in f:
            s = json.loads(line)
            image_id = s["image_id"]
            if image_id in seen:
                continue
            seen.add(image_id)

            p = find_image(image_id)
            if p is None:
                missing += 1
                continue

            with Image.open(p) as img:
                dims[img.size] += 1

    print(f"\n{split}")
    print("unique images:", len(seen))
    print("missing images:", missing)
    print("top dimensions:")
    for (w, h), c in dims.most_common(20):
        print(f"  {w}x{h}: {c}")