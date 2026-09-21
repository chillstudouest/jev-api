"""Encoder backbone with a marker-pointer head and a separate calibrated correctness head.

Faithful copy of upstream openJev-verdict-2.0 `verdict2/model.py`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

NEG = -1e4
CORRECTNESS_FEATURES = 7  # p1, margin, normalized entropy, 1/K, one-hot(qtype)


class CorrectnessHead(nn.Module):
    """Predicts the probability that the pointer head's argmax equals the gold label."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CORRECTNESS_FEATURES, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    @staticmethod
    def features(probs: torch.Tensor, marker_mask: torch.Tensor, qtype: torch.Tensor) -> torch.Tensor:
        """Build shape features from a probability distribution. Detached by design."""
        p = probs.detach().float()
        k = marker_mask.sum(-1).clamp(min=2).float()
        top2 = p.topk(2, dim=-1).values
        entropy = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        onehot = torch.zeros(p.size(0), 3, device=p.device, dtype=p.dtype)
        onehot.scatter_(1, qtype.view(-1, 1), 1.0)
        return torch.cat(
            [top2[:, :1], (top2[:, :1] - top2[:, 1:2]), entropy.unsqueeze(-1), (1.0 / k).unsqueeze(-1), onehot],
            dim=-1,
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)


class VerdictModel(nn.Module):
    """Bidirectional encoder plus per-option marker scoring."""

    def __init__(self, backbone: str = "answerdotai/ModernBERT-base", dropout: float = 0.1) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        # Prefer local/config-only init so the Docker image never needs to fetch
        # ModernBERT weights at build time. Fine-tuned encoder weights arrive via
        # load_state_dict from the Verdict 2.0 checkpoint.
        config = AutoConfig.from_pretrained(backbone)
        try:
            self.encoder = AutoModel.from_config(config, attn_implementation="sdpa")
        except (TypeError, ValueError):
            self.encoder = AutoModel.from_config(config)
        d = self.encoder.config.hidden_size
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
        )
        self.correctness = CorrectnessHead()
        # one temperature per (qtype, cardinality bucket); filled in by calibrate.py
        self.register_buffer("temperature", torch.ones(3, 8))

    def option_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return [B, Kmax] logits, with padded option slots masked out."""
        hidden = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        hidden = hidden + self.type_emb(batch["qtype"])[:, None, :]

        index = batch["marker_pos"].clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        marker_hidden = torch.gather(hidden, 1, index)

        logits = self.scorer(marker_hidden).squeeze(-1).float()
        return logits.masked_fill(~batch["marker_mask"], NEG)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.option_logits(batch)
        probs = torch.softmax(logits, dim=-1)
        feats = CorrectnessHead.features(probs, batch["marker_mask"], batch["qtype"])
        return logits, self.correctness(feats)


def cardinality_bucket(k: int) -> int:
    """Map option count to a temperature bucket index in [0, 7]."""
    return min(max(k, 2), 9) - 2


def apply_temperature(
    logits: torch.Tensor, qtype: torch.Tensor, k: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    """Divide logits by the fitted per-(qtype, cardinality) temperature."""
    buckets = torch.clamp(k, 2, 9).long() - 2
    t = temperature[qtype.long(), buckets].clamp(min=0.05).unsqueeze(-1)
    return logits / t
