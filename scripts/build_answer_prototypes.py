#!/usr/bin/env python3
"""
Generate offline projected prototypes for COCO-QA object and color answers.
Supports sentence-transformers embeddings with deterministic random projection,
or falls back to a deterministic text-hash vector generator if sentence-transformers is absent.
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_deterministic_fallback(text: str, embed_dim: int = 128) -> torch.Tensor:
    """
    Generates a deterministic normalized unit vector of size embed_dim
    based on the SHA-256 hash of the input text string.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], byteorder="big")
    g = torch.Generator()
    g.manual_seed(seed)
    vec = torch.randn(embed_dim, generator=g)
    return vec / vec.norm().clamp(min=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-vocab", type=Path, default=Path("data/processed/answer_vocab.json"))
    parser.add_argument("--pt-out", type=Path, default=Path("data/processed/answer_prototypes.pt"))
    parser.add_argument("--json-out", type=Path, default=Path("data/processed/answer_prototypes.json"))
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=42)

    args = parser.parse_args()

    print("--- VQA Prototype Builder ---")
    if not args.answer_vocab.exists():
        print(f"ERROR: answer_vocab.json not found at {args.answer_vocab}")
        sys.exit(1)

    with args.answer_vocab.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    object_classes = vocab["object_answers"]
    color_classes = vocab["color_answers"]

    print(f"Loaded {len(object_classes)} object classes and {len(color_classes)} color classes.")

    # 1. Create Prompts
    object_prompts = [f"a photo of a {c}" for c in object_classes]
    color_prompts = [f"the color {c}" for c in color_classes]

    print("\nExample prompts:")
    print(f"  Object: '{object_prompts[0]}' -> class: '{object_classes[0]}'")
    print(f"  Color:  '{color_prompts[0]}'  -> class: '{color_classes[0]}'")

    # 2. Embedding Generation
    sentence_transformers_available = False
    try:
        from sentence_transformers import SentenceTransformer
        sentence_transformers_available = True
    except ImportError:
        pass

    if sentence_transformers_available:
        print("\n================================================================================")
        print(">>> EMBEDDING BACKEND: sentence-transformers (REAL TEXT EMBEDDINGS) <<<")
        print("================================================================================")
        print("[INFO] sentence-transformers package detected. Loading 'all-MiniLM-L6-v2'...")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            print("Successfully loaded model. Generating text embeddings...")

            raw_obj_embs = torch.tensor(model.encode(object_prompts), dtype=torch.float32)
            raw_col_embs = torch.tensor(model.encode(color_prompts), dtype=torch.float32)

            raw_dim = raw_obj_embs.shape[1]
            print(f"Generated raw embeddings of dimension {raw_dim}.")

            # Deterministic projection to embed_dim (128)
            print(f"Projecting raw {raw_dim}-dim embeddings to {args.embed_dim}-dim via deterministic projection...")
            g = torch.Generator()
            g.manual_seed(args.projection_seed)
            proj_matrix = torch.randn(raw_dim, args.embed_dim, generator=g)

            obj_prototypes = torch.matmul(raw_obj_embs, proj_matrix)
            color_prototypes = torch.matmul(raw_col_embs, proj_matrix)

            # L2-normalize
            obj_prototypes = F.normalize(obj_prototypes, p=2, dim=1)
            color_prototypes = F.normalize(color_prototypes, p=2, dim=1)

        except Exception as e:
            print(f"WARNING: sentence-transformers failed with error: {e}. Falling back to deterministic hashes.")
            sentence_transformers_available = False

    if not sentence_transformers_available:
        print("\n================================================================================")
        print(">>> EMBEDDING BACKEND: fallback hash embeddings (DETERMINISTIC FALLBACK) <<<")
        print("================================================================================")
        print("[WARNING] sentence-transformers is unavailable locally. Using stable, deterministic SHA-256 hash generator.")
        model_name = "deterministic-hash-sha256"

        obj_list = [get_deterministic_fallback(p, args.embed_dim) for p in object_prompts]
        color_list = [get_deterministic_fallback(p, args.embed_dim) for p in color_prompts]

        obj_prototypes = torch.stack(obj_list)
        color_prototypes = torch.stack(color_list)

    # Asserts to verify properties
    assert obj_prototypes.shape == (len(object_classes), args.embed_dim)
    assert color_prototypes.shape == (len(color_classes), args.embed_dim)

    # Check normalization
    assert torch.allclose(torch.norm(obj_prototypes, p=2, dim=1), torch.ones(len(object_classes)), atol=1e-5)
    assert torch.allclose(torch.norm(color_prototypes, p=2, dim=1), torch.ones(len(color_classes)), atol=1e-5)

    print("\nAll prototypes successfully generated, projected, and L2-normalized.")
    print(f"Object prototypes shape: {list(obj_prototypes.shape)}")
    print(f"Color prototypes shape:  {list(color_prototypes.shape)}")

    # 3. Save files
    print(f"\nSaving PyTorch weights file to {args.pt_out}...")
    args.pt_out.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict_pt = {
        "object_prototypes": obj_prototypes,
        "color_prototypes": color_prototypes,
        "object_classes": object_classes,
        "color_classes": color_classes,
        "text_embedding_model_name": model_name,
        "answer_embed_dim": args.embed_dim,
        "object_prompts": object_prompts,
        "color_prompts": color_prompts,
    }
    torch.save(save_dict_pt, args.pt_out)

    print(f"Saving JSON config file to {args.json_out}...")
    save_dict_json = {
        "object_prototypes": obj_prototypes.tolist(),
        "color_prototypes": color_prototypes.tolist(),
        "object_classes": object_classes,
        "color_classes": color_classes,
        "text_embedding_model_name": model_name,
        "answer_embed_dim": args.embed_dim,
        "object_prompts": object_prompts,
        "color_prompts": color_prompts,
    }
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(save_dict_json, f, indent=2, ensure_ascii=False)

    print("Success. Offline prototype generation complete!")


if __name__ == "__main__":
    main()
