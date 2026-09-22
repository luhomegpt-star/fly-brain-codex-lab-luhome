from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, math
import numpy as np
from .semantic import DIMENSIONS, SemanticVector

@dataclass
class FlyResult:
    backend: str
    stimulated_neurons: list[int]
    active_neurons: int
    total_spikes: int
    top_firing_neurons: list[dict]
    output_activity: dict[str, float]
    neural_activity_summary: dict[str, float]
    def to_dict(self): return asdict(self)

class ProxyFlyBrain:
    size = 166700
    def __init__(self, seed=5175): self.seed = seed
    def _indices(self, dimension, n=24):
        h = hashlib.sha256(f'{self.seed}:{dimension}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], 'big'))
        return sorted(rng.choice(self.size, size=n, replace=False).tolist())
    def run(self, vector: SemanticVector, steps=64):
        values = vector.to_dict(); activity = {}; stimulated = []
        for d in DIMENSIONS:
            for rank, idx in enumerate(self._indices(d)):
                amp = values[d] * (1 - rank / (24 * 1.8))
                stimulated.append(idx); activity[idx] = activity.get(idx, 0.0) + amp
        ordered = sorted(activity.items(), key=lambda kv: kv[1], reverse=True)
        active = sum(v > 0.15 for _, v in ordered)
        spikes = int(sum(max(0.0, v - 0.08) * steps * 1.7 for _, v in ordered))
        total = sum(activity.values()) or 1.0
        entropy = -sum((v/total)*math.log((v/total)+1e-12) for _, v in ordered)
        out = {
            'approach_like': round(0.65*values['approach']+0.35*values['reward'], 6),
            'avoidance_like': round(0.65*values['avoidance']+0.35*values['threat'], 6),
            'high_arousal_like': round(0.55*values['arousal']+0.25*values['motion']+0.20*values['novelty'], 6),
            'uncertainty_like': round(values['uncertainty'], 6),
        }
        return FlyResult('proxy', stimulated, active, spikes,
            [{'neuron_id': i, 'activity': round(float(v), 6)} for i, v in ordered[:12]],
            out, {'activity_entropy_proxy': round(float(entropy), 6)})

class FlyBrainBackend:
    def __init__(self, seed=5175):
        try:
            import flybrain
        except ImportError as exc:
            raise RuntimeError('Install optional brain dependencies first') from exc
        self.flybrain = flybrain; self.seed = seed
    def run(self, vector: SemanticVector, steps=64):
        raise NotImplementedError('Real flybrain adapter requires local API validation before use.')

def get_backend(name, seed=5175):
    if name == 'proxy': return ProxyFlyBrain(seed)
    if name == 'flybrain': return FlyBrainBackend(seed)
    raise ValueError(name)
