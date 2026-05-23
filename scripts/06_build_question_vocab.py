#!/usr/bin/env python3
"""
Build question vocabulary from the training split only.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.text import build_question_vocab, question_lengths, tokenize


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


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0

    values_sorted = sorted(values)
    idx = int(round((p / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[idx]


def build_stats(samples: list[dict], vocab, lengths: list[int]) -> dict:
    token_counter = Counter()

    for sample in samples:
        token_counter.update(tokenize(sample["question"]))

    num_tokens_total = sum(token_counter.values())
    num_tokens_in_vocab = 0

    for token, count in token_counter.items():
        if token in vocab.token_to_id:
            num_tokens_in_vocab += count

    coverage = (
        num_tokens_in_vocab / num_tokens_total
        if num_tokens_total > 0
        else 0.0
    )

    return {
        "num_train_questions": len(samples),
        "vocab_size": vocab.size,
        "max_length": vocab.max_length,
        "num_unique_tokens_before_pruning": len(token_counter),
        "num_tokens_total": num_tokens_total,
        "token_coverage": coverage,
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": sum(lengths) / len(lengths) if lengths else 0,
            "p50": percentile(lengths, 50),
            "p90": percentile(lengths, 90),
            "p95": percentile(lengths, 95),
            "p99": percentile(lengths, 99),
        },
        "top_50_tokens": token_counter.most_common(50),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/processed/cocoqa_train_resolved.jsonl"),
    )
    parser.add_argument(
        "--vocab-out",
        type=Path,
        default=Path("data/processed/question_vocab.json"),
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=Path("data/processed/question_vocab_stats.json"),
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=24,
    )

    args = parser.parse_args()

    print(f"Loading train manifest: {args.train_manifest}")
    samples = read_jsonl(args.train_manifest)

    questions = [sample["question"] for sample in samples]

    print("Building question vocabulary from train questions only...")
    vocab = build_question_vocab(
        questions=questions,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        max_length=args.max_length,
    )

    lengths = question_lengths(questions)
    stats = build_stats(samples, vocab, lengths)

    print(f"Saving vocab: {args.vocab_out}")
    vocab.save(args.vocab_out)

    print(f"Saving stats: {args.stats_out}")
    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()