from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import random
import re
from typing import Mapping

DIMENSIONS = (
    "reward",
    "threat",
    "novelty",
    "arousal",
    "approach",
    "avoidance",
    "motion",
    "uncertainty",
)


@dataclass(frozen=True)
class SemanticVector:
    reward: float
    threat: float
    novelty: float
    arousal: float
    approach: float
    avoidance: float
    motion: float
    uncertainty: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "SemanticVector":
        missing = [name for name in DIMENSIONS if name not in values]
        if missing:
            raise ValueError(f"Missing semantic dimensions: {missing}")
        cleaned = {}
        for name in DIMENSIONS:
            value = float(values[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
            cleaned[name] = value
        return cls(**cleaned)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


_LEXICONS: dict[str, tuple[str, ...]] = {
    "reward": (
        "reward", "profit", "gain", "good", "food", "sugar", "sweet",
        "獎勵", "獲利", "上漲", "甜", "糖", "食物", "好消息",
    ),
    "threat": (
        "threat", "danger", "crash", "panic", "attack", "loss",
        "威脅", "危險", "暴跌", "恐慌", "攻擊", "虧損", "黑天鵝",
    ),
    "novelty": (
        "new", "novel", "unexpected", "surprise", "unknown",
        "新", "首次", "意外", "驚訝", "陌生",
    ),
    "arousal": (
        "urgent", "fast", "huge", "extreme", "volatile", "explosive",
        "緊急", "快速", "巨大", "極端", "劇烈", "爆量", "高波動",
    ),
    "approach": (
        "approach", "buy", "enter", "chase", "toward",
        "接近", "買入", "進場", "追", "靠近",
    ),
    "avoidance": (
        "avoid", "escape", "sell", "exit", "run",
        "避開", "逃跑", "賣出", "出場", "遠離",
    ),
    "motion": (
        "move", "moving", "speed", "trend", "breakout",
        "移動", "速度", "趨勢", "突破", "急拉", "急殺",
    ),
    "uncertainty": (
        "maybe", "uncertain", "unknown", "conflict", "mixed",
        "可能", "不確定", "未知", "矛盾", "混合", "盤整",
    ),
}


def keyword_encode(text: str) -> SemanticVector:
    """Deterministic, transparent baseline encoder.

    This is intentionally simple. It is a baseline for comparison with Codex,
    not a claim that keyword counts recover meaning.
    """
    normalized = text.lower()
    scores: dict[str, float] = {}
    token_count = max(1, len(re.findall(r"\w+|[\u4e00-\u9fff]", normalized)))

    for dimension, words in _LEXICONS.items():
        hits = sum(normalized.count(word.lower()) for word in words)
        # Smoothly saturate; short text with one hit should be noticeable,
        # repeated hits should not exceed 1.
        raw = hits / max(1.0, token_count ** 0.35)
        scores[dimension] = min(1.0, 0.12 + 0.55 * raw) if hits else 0.12

    # Small transparent coupling to avoid eight entirely independent bins.
    scores["arousal"] = min(1.0, scores["arousal"] + 0.20 * scores["threat"])
    scores["avoidance"] = min(1.0, scores["avoidance"] + 0.18 * scores["threat"])
    scores["approach"] = min(1.0, scores["approach"] + 0.12 * scores["reward"])
    return SemanticVector.from_mapping(scores)


def random_encode(text: str, seed: int = 0) -> SemanticVector:
    """Deterministic random-control encoder for the same text and seed."""
    digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return SemanticVector.from_mapping({name: rng.random() for name in DIMENSIONS})
