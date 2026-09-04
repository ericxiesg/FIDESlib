"""A BERT WordPiece tokenizer, in numpy-free plain Python.

``transformers`` is not installed on the benchmark machine, and installing it would pull torch in for
what is, for BERT, a well-specified hundred lines: normalise, split on whitespace and punctuation,
then greedy longest-match-first against the vocabulary with ``##`` marking continuations. This is a
transcription of ``BasicTokenizer`` + ``WordpieceTokenizer`` from the original BERT release, which is
what ``BertTokenizer`` still implements.

Anything that changes tokenization changes accuracy, so :func:`BertTokenizer.from_files` reads the
model's own ``tokenizer_config.json`` for ``do_lower_case`` rather than assuming the uncased default.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path


def _is_whitespace(char):
    return char in " \t\n\r" or unicodedata.category(char) == "Zs"


def _is_control(char):
    return char not in "\t\n\r" and unicodedata.category(char).startswith("C")


def _is_punctuation(char):
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def _is_chinese(code):
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF or 0xF900 <= code <= 0xFAFF
            or 0x2F800 <= code <= 0x2FA1F)


class BertTokenizer:
    """WordPiece over a ``vocab.txt``. ``do_lower_case`` follows the checkpoint's own config."""

    def __init__(self, vocab: dict[str, int], *, do_lower_case: bool = True,
                 unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
                 max_word_chars: int = 100):
        self.vocab = vocab
        self.do_lower_case = do_lower_case
        self.unk, self.cls, self.sep, self.pad = unk_token, cls_token, sep_token, pad_token
        self.max_word_chars = max_word_chars
        for token in (unk_token, cls_token, sep_token, pad_token):
            if token not in vocab:
                raise ValueError(f"vocabulary is missing {token}")

    @classmethod
    def from_files(cls, vocab_path, config_path=None) -> "BertTokenizer":
        vocab = {}
        with open(vocab_path, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                vocab[line.rstrip("\n")] = index

        lower = True
        if config_path is not None and Path(config_path).exists():
            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
            lower = bool(config.get("do_lower_case", True))
        return cls(vocab, do_lower_case=lower)

    # ------------------------------------------------------------------ basic
    def _clean(self, text):
        out = []
        for char in text:
            code = ord(char)
            if code == 0 or code == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                out.append(" ")
            elif _is_chinese(code):
                out.append(f" {char} ")
            else:
                out.append(char)
        return "".join(out)

    def _strip_accents(self, text):
        return "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")

    def _split_punctuation(self, word):
        pieces, current = [], []
        for char in word:
            if _is_punctuation(char):
                if current:
                    pieces.append("".join(current))
                    current = []
                pieces.append(char)
            else:
                current.append(char)
        if current:
            pieces.append("".join(current))
        return pieces

    def basic_tokenize(self, text):
        tokens = []
        for word in self._clean(text).split():
            if self.do_lower_case:
                word = self._strip_accents(word.lower())
            tokens.extend(self._split_punctuation(word))
        return tokens

    # ------------------------------------------------------------------ wordpiece
    def wordpiece(self, word):
        if len(word) > self.max_word_chars:
            return [self.unk]
        pieces, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                return [self.unk]
            pieces.append(found)
            start = end
        return pieces

    def tokenize(self, text):
        return [piece for word in self.basic_tokenize(text) for piece in self.wordpiece(word)]

    # ------------------------------------------------------------------ encoding
    def encode_pair(self, first: str, second: str | None = None, *, max_length: int = 128):
        """``[CLS] a [SEP] b [SEP]`` with HuggingFace's ``longest_first`` truncation.

        Returns ``(input_ids, token_type_ids, attention_mask)``, each of length ``max_length``.
        """
        tokens_a = self.tokenize(first)
        tokens_b = self.tokenize(second) if second is not None else None

        if tokens_b is None:
            del tokens_a[max_length - 2:]
        else:
            budget = max_length - 3
            while len(tokens_a) + len(tokens_b) > budget:
                if len(tokens_a) >= len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()

        tokens = [self.cls, *tokens_a, self.sep]
        types = [0] * len(tokens)
        if tokens_b is not None:
            tokens += [*tokens_b, self.sep]
            types += [1] * (len(tokens_b) + 1)

        ids = [self.vocab.get(token, self.vocab[self.unk]) for token in tokens]
        mask = [1] * len(ids)

        padding = max_length - len(ids)
        if padding < 0:
            raise ValueError("truncation failed to fit the sequence")
        ids += [self.vocab[self.pad]] * padding
        types += [0] * padding
        mask += [0] * padding
        return ids, types, mask
