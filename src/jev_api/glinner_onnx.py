"""Torch-free GLiNER2.5-Decide classification: onnxruntime + tokenizers + numpy.

Port of `gliner_onnx.py` from nishparadox/gliner2.5-decide-onnx (Apache-2.0),
an export of the classification path of fastino/GLiNER2.5-Decide.

Rebuilds exactly what gliner2's processor feeds the encoder (inference mode):
    ( [P] "{task} [DESCRIPTION] {label}: {desc} ..." ( [L] l1 [L] l2 ... ) )
    [SEP_STRUCT] <next task> ... [SEP_TEXT] <lowercased words>
Each piece is tokenized on its own, with no [CLS]/[SEP]. Label logits are
read at the [L] markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# DeBERTa-v3 position limit.
MAX_TOKENS = 512

_WORDS = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


@dataclass
class OnnxTask:
    name: str
    labels: dict[str, str | None]

    def pieces(self) -> list[str]:
        prompt = self.name
        for label, desc in self.labels.items():
            if desc:
                prompt += f" [DESCRIPTION] {label}: {desc}"
        pieces = ["(", "[P]", prompt, "("]
        for label in self.labels:
            pieces += ["[L]", label]
        return pieces + [")", ")"]


class GlinerOnnx:
    def __init__(self, model_path: Path, tokenizer_path: Path, threads: int | None = None) -> None:
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._cache: dict[str, list[int]] = {}

    def _ids(self, piece: str) -> list[int]:
        ids = self._cache.get(piece)
        if ids is None:
            ids = self.tokenizer.encode(piece, add_special_tokens=False).ids
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, tasks: list[OnnxTask]) -> tuple[list[int], list[int]]:
        if not text.endswith((".", "!", "?")):
            text = f"{text}." if text else "."
        ids: list[int] = []
        positions: list[int] = []
        for index, task in enumerate(tasks):
            if index:
                ids += self._ids("[SEP_STRUCT]")
            pieces = task.pieces()
            for position, piece in enumerate(pieces):
                if piece == "[L]" and position >= 4 and position % 2 == 0 and position < len(pieces) - 2:
                    positions.append(len(ids))
                ids += self._ids(piece)
        ids += self._ids("[SEP_TEXT]")
        # Words are tokenized one at a time and never cached: the state changes per request.
        for match in _WORDS.finditer(text):
            word_ids = self.tokenizer.encode(match.group().lower(), add_special_tokens=False).ids
            if len(ids) + len(word_ids) > MAX_TOKENS:
                break
            ids += word_ids
        if len(ids) > MAX_TOKENS:
            raise ValueError(f"Questions alone exceed {MAX_TOKENS} tokens")
        return ids, positions

    def probabilities(self, text: str, tasks: list[OnnxTask]) -> dict[str, dict[str, float]]:
        ids, positions = self.encode(text, tasks)
        (logits,) = self.session.run(
            ["logits"],
            {
                "input_ids": np.asarray([ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
                "label_positions": np.asarray([positions], dtype=np.int64),
            },
        )
        flat = iter(logits[0].tolist())
        result: dict[str, dict[str, float]] = {}
        for task in tasks:
            values = np.array([next(flat) for _ in task.labels], dtype=np.float64)
            exps = np.exp(values - values.max())
            result[task.name] = dict(zip(task.labels, (exps / exps.sum()).tolist()))
        return result
