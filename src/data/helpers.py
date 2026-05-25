import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    return samples


def read_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_answer_vocab(path: str | Path) -> dict:
    data = read_json(path)

    # Reconcile different formats: some scripts expect global id_to_answer, some expect full dict
    return data


def id_to_answer_from_vocab(answer_vocab: dict) -> dict[int, str]:
    if "id_to_answer" in answer_vocab:
        return {int(k): v for k, v in answer_vocab["id_to_answer"].items()}
    return {int(k): v for k, v in answer_vocab.items()}


def build_answer_ids_by_type(processed_dir: str | Path) -> dict[str, list[int]]:
    """
    Build type-specific answer ID sets from resolved manifests.
    """
    processed_dir = Path(processed_dir)
    paths = [
        processed_dir / "cocoqa_train_resolved.jsonl",
        processed_dir / "cocoqa_val_resolved.jsonl",
        processed_dir / "cocoqa_test_resolved.jsonl",
    ]

    answer_ids_by_type = {
        "object": set(),
        "color": set(),
        "number": set(),
    }

    for path in paths:
        if not path.exists():
            continue

        for sample in read_jsonl(path):
            qtype = sample["type"]
            if qtype not in answer_ids_by_type:
                continue

            answer_ids_by_type[qtype].add(int(sample["answer_id"]))

    out = {
        qtype: sorted(ids)
        for qtype, ids in answer_ids_by_type.items()
    }

    for qtype, ids in out.items():
        if not ids and qtype != "number":  # number might be absent in object/color datasets
            raise ValueError(f"No answer IDs found for question type: {qtype}")

    return out


def build_answer_ids_from_vocab(answer_vocab: dict) -> dict[str, list[int]]:
    """
    Build answer IDs per question type using the metadata in answer_vocab.json.
    """
    if "answer_to_id" not in answer_vocab:
        # Fallback to scanning manifests if vocab lacks answer_to_id metadata
        return build_answer_ids_by_type(Path("data/processed"))

    answer_to_id = answer_vocab["answer_to_id"]

    object_answer_ids = [
        int(answer_to_id[ans])
        for ans in answer_vocab.get("object_answers", [])
        if ans in answer_to_id
    ]

    color_answer_ids = [
        int(answer_to_id[ans])
        for ans in answer_vocab.get("color_answers", [])
        if ans in answer_to_id
    ]

    number_answer_ids = [
        int(answer_to_id[ans])
        for ans in answer_vocab.get("number_answers", [])
        if ans in answer_to_id
    ]

    return {
        "object": sorted(object_answer_ids),
        "color": sorted(color_answer_ids),
        "number": sorted(number_answer_ids),
    }
