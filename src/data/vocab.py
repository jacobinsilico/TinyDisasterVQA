import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for COCO-QA questions.

    Example:
      "What is the color of the dog?"
      -> ["what", "is", "the", "color", "of", "the", "dog"]
    """
    text = text.lower().strip()
    return _TOKEN_PATTERN.findall(text)


@dataclass
class QuestionVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    max_length: int

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, question: str) -> tuple[list[int], int]:
        """
        Encode a question into fixed-length token IDs.

        Returns:
          token_ids: list[int] of length max_length
          length: original length after truncation, before padding
        """
        tokens = tokenize(question)
        tokens = tokens[: self.max_length]

        length = len(tokens)

        token_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]

        if len(token_ids) < self.max_length:
            token_ids += [self.pad_id] * (self.max_length - len(token_ids))

        return token_ids, length

    def decode(self, token_ids: Iterable[int], remove_special: bool = True) -> str:
        """
        Decode token IDs back into a rough text string.
        """
        tokens = []

        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), UNK_TOKEN)

            if remove_special and token in {PAD_TOKEN}:
                continue

            tokens.append(token)

        return " ".join(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},
            "max_length": self.max_length,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: str | Path) -> "QuestionVocab":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        token_to_id = {
            str(token): int(idx)
            for token, idx in data["token_to_id"].items()
        }

        id_to_token = {
            int(idx): str(token)
            for idx, token in data["id_to_token"].items()
        }

        return QuestionVocab(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            max_length=int(data["max_length"]),
        )


def build_question_vocab(
    questions: Iterable[str],
    max_vocab_size: int = 2000,
    min_freq: int = 1,
    max_length: int = 24,
) -> QuestionVocab:
    """
    Build question vocabulary from training questions only.

    Args:
      questions:
        Iterable of raw question strings.
      max_vocab_size:
        Maximum vocab size including <pad> and <unk>.
      min_freq:
        Minimum token frequency needed to enter vocabulary.
      max_length:
        Fixed encoded question length.

    Returns:
      QuestionVocab
    """
    if max_vocab_size < 2:
        raise ValueError("max_vocab_size must be at least 2.")

    counter = Counter()

    for question in questions:
        counter.update(tokenize(question))

    token_to_id = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    # Reserve 2 slots for <pad> and <unk>.
    num_normal_tokens = max_vocab_size - 2

    for token, freq in counter.most_common():
        if freq < min_freq:
            continue

        if len(token_to_id) >= max_vocab_size:
            break

        token_to_id[token] = len(token_to_id)

    id_to_token = {idx: token for token, idx in token_to_id.items()}

    return QuestionVocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        max_length=max_length,
    )


def question_lengths(questions: Iterable[str]) -> list[int]:
    """
    Return tokenized question lengths. Useful for inspecting max_length choice.
    """
    return [len(tokenize(question)) for question in questions]
