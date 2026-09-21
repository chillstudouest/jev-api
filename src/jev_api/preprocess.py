"""Dataset loading, sequence construction for typed-decisions.

Faithful copy of upstream openJev-verdict-2.0 `verdict2/data.py` (inference-relevant parts).

Sequence layout:
    [CLS] {qtype} question: {instructions} [SEP] [MASK]{opt0} [MASK]{opt1} ... [SEP] {state} [SEP]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

QTYPES = {"choice": 0, "score": 1, "noul": 2}

# API alias: "workflow" is a choice among action options.
QTYPE_ALIASES = {
    "workflow": "choice",
    "choice": "choice",
    "score": "score",
    "noul": "noul",
}

NOUL_DEFAULT = {
    "false": "no, the statement does not hold",
    "true": "yes, the statement holds",
}


@dataclass(frozen=True)
class Item:
    """One typed question, ready for the encoder."""

    case_id: str
    workflow: str
    qid: str
    qtype: int
    ids: Tuple[int, ...]
    markers: Tuple[int, ...]
    target: Tuple[float, ...]
    label: int
    option_keys: Tuple[str, ...]
    gold_score: float
    source: Dict[str, Any] | None = None


def render_options(qtype: str, criteria: Any) -> Tuple[List[str], List[str]]:
    """Return (rendered option texts, option keys) in a stable, label-aligned order."""
    if qtype == "choice":
        keys = list(criteria.keys())
        return [f"{k}: {criteria[k]}" if criteria[k] else k for k in keys], keys
    if qtype == "score":
        return [f"level {i}: {c}" for i, c in enumerate(criteria)], [str(i) for i in range(len(criteria))]
    crit = criteria or {}
    keys = ["false", "true"]
    return [f"{k}: {crit.get(k) or NOUL_DEFAULT[k]}" for k in keys], keys


def build_item(
    tokenizer: Any,
    row: Dict[str, Any],
    qid: str,
    qdef: Dict[str, Any],
    gold: Dict[str, Any],
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Sequence[int] | None = None,
) -> Item | None:
    """Tokenize one question. Returns None when the option markers do not fit in max_len."""
    qtype = qdef["type"]
    texts, keys = render_options(qtype, qdef.get("criteria"))
    order = list(option_order) if option_order is not None else list(range(len(texts)))

    mask_tok = tokenizer.mask_token
    instructions = str(qdef["instructions"]).replace(mask_tok, " ")

    head_ids = tokenizer(f"{qtype} question: {instructions}", add_special_tokens=False)["input_ids"]
    opt_ids: List[List[int]] = []
    for i in order:
        body = tokenizer(" " + texts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        opt_ids.append([tokenizer.mask_token_id] + body)

    budget = head_max_len - sum(len(o) for o in opt_ids)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, budget)]

    ids: List[int] = [tokenizer.cls_token_id] + head_ids + [tokenizer.sep_token_id]
    markers: List[int] = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tokenizer.sep_token_id)

    room = max(0, max_len - len(ids) - 1)
    state = tokenizer(str(row["state"]).replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:room]
    ids = (ids + state + [tokenizer.sep_token_id])[:max_len]

    if any(m >= max_len for m in markers):
        return None

    probs = gold.get("probabilities", {})
    raw = [float(probs.get(keys[i], 0.0)) for i in order]
    total = sum(raw)
    target = [p / total for p in raw] if total > 0 else [1.0 / len(raw)] * len(raw)

    ordered_keys = [keys[i] for i in order]
    gold_label = str(gold.get("label", ""))
    label = (
        ordered_keys.index(gold_label)
        if gold_label in ordered_keys
        else int(max(range(len(target)), key=target.__getitem__))
    )

    return Item(
        case_id=str(row.get("id", "")),
        workflow=str(row.get("workflow", "")),
        qid=qid,
        qtype=QTYPES[qtype],
        ids=tuple(ids),
        markers=tuple(markers),
        target=tuple(target),
        label=label,
        option_keys=tuple(ordered_keys),
        gold_score=float(gold.get("score", 0.0)),
        source={
            "state": str(row["state"]),
            "qdef": qdef,
            "gold": gold,
            "id": row.get("id", ""),
            "workflow": row.get("workflow", ""),
        },
    )


def collate(batch: List[Item], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a batch of Items into dense tensors."""
    n = len(batch)
    length = max(len(it.ids) for it in batch)
    kmax = max(len(it.markers) for it in batch)

    ids = torch.full((n, length), pad_id, dtype=torch.long)
    attn = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)

    for i, it in enumerate(batch):
        ids[i, : len(it.ids)] = torch.tensor(it.ids, dtype=torch.long)
        attn[i, : len(it.ids)] = 1
        k = len(it.markers)
        mpos[i, :k] = torch.tensor(it.markers, dtype=torch.long)
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it.target, dtype=torch.float32)

    return {
        "input_ids": ids,
        "attention_mask": attn,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "target": target,
        "label": torch.tensor([it.label for it in batch], dtype=torch.long),
        "qtype": torch.tensor([it.qtype for it in batch], dtype=torch.long),
    }


def resolve_qtype(raw: str) -> str:
    key = raw.strip().lower()
    if key not in QTYPE_ALIASES:
        raise ValueError(f"Unsupported type '{raw}'. Expected one of: {sorted(QTYPE_ALIASES)}")
    return QTYPE_ALIASES[key]


def build_decision_item(
    tokenizer: Any,
    *,
    qtype_raw: str,
    question: str,
    state: str,
    options: List[Dict[str, str]],
    case_id: str = "",
    workflow: str = "",
    qid: str = "decide",
) -> Item:
    """Build an Item for live API inference using the exact upstream tokenization path."""
    qtype = resolve_qtype(qtype_raw)

    if qtype == "noul":
        criteria: Any = {o["id"]: o.get("description", "") for o in options} if options else {}
    elif qtype == "score":
        # Preserve caller order; criteria is a list of level descriptions.
        criteria = [o.get("description") or o["id"] for o in options]
    else:
        if len(options) < 2:
            raise ValueError("choice/workflow decisions require at least 2 options")
        criteria = {o["id"]: o.get("description", "") for o in options}

    qdef = {"type": qtype, "instructions": question, "criteria": criteria}
    keys = render_options(qtype, criteria)[1]
    gold = {
        "probabilities": {k: 0.0 for k in keys},
        "label": keys[0] if keys else "",
        "score": 0.0,
    }
    row = {"state": state, "id": case_id, "workflow": workflow}
    item = build_item(tokenizer, row, qid, qdef, gold)
    if item is None:
        raise ValueError("Decision sequence exceeds max_len=512 (markers past context window)")
    return item
