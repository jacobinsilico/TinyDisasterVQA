"""
Backwards compatible shell for text.
"""

from src.data.vocab import (
    PAD_TOKEN,
    UNK_TOKEN,
    tokenize,
    QuestionVocab,
    build_question_vocab,
    question_lengths,
)