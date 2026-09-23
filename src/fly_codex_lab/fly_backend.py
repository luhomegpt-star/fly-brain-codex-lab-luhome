from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
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
    neural_activity_summary: dict

    def to_dict(self):
        return asdict(self)


class ProxyFlyBrain:
    """Synthetic plumbing check only. This is not a biological simulation."""

    size = 166700

    def __init__(self, seed=5175):
        self.seed = seed

    def _indices(self, dimension, n=24):
        h = hashlib.sha256(f"{self.seed}:{dimension}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        return sorted(rng.choice(self.size, size=n, replace=False).tolist())

    def run(self, vector: SemanticVector, steps=64):
        values = vector.to_dict()
        activity = {}
        stimulated = []

        for d in DIMENSIONS:
            for rank, idx in enumerate(self._indices(d)):
                amp = values[d] * (1 - rank / (24 * 1.8))
                stimulated.append(idx)
                activity[idx] = activity.get(idx, 0.0) + amp

        ordered = sorted(activity.items(), key=lambda kv: kv[1], reverse=True)
        active = sum(v > 0.15 for _, v in ordered)
        spikes = int(sum(max(0.0, v - 0.08) * steps * 1.7 for _, v in ordered))
        total = sum(activity.values()) or 1.0
        entropy = -sum((v / total) * math.log((v / total) + 1e-12) for _, v in ordered)

        out = {
            "approach_like": round(0.65 * values["approach"] + 0.35 * values["reward"], 6),
            "avoidance_like": round(0.65 * values["avoidance"] + 0.35 * values["threat"], 6),
            "high_arousal_like": round(
                0.55 * values["arousal"] + 0.25 * values["motion"] + 0.20 * values["novelty"],
                6,
            ),
            "uncertainty_like": round(values["uncertainty"], 6),
        }

        return FlyResult(
            "proxy",
            stimulated,
            active,
            spikes,
            [{"neuron_id": i, "activity": round(float(v), 6)} for i, v in ordered[:12]],
            out,
            {"activity_entropy_proxy": round(float(entropy), 6)},
        )


class FlyBrainBackend:
    """Real MaleCNS connectome backend using flybrain 0.1.x.

    IMPORTANT:
    The semantic-vector -> visual-neuron mapping below is an experimental
    encoder chosen by us. The connectome itself is real/fixed; the encoder is
    not a claim that a fly understands words such as "threat" or "reward".
    """

    # Cell types documented by fly.ai as visual feature detectors.
    INPUT_TYPES = {
        "looming_escape": ["LC4", "LPLC2"],
        "small_approach": ["LPLC1"],
        "chase_target": ["LC10a"],
    }

    # Identified descending/command neurons documented by fly.ai.
    COMMAND_TYPES = {
        "escape_takeoff_DNp01": ["DNp01"],
        "steering_DNa02": ["DNa02"],
        "forward_walk_DNg100": ["DNg100"],
        "backward_walk_MDN": ["MDN"],
    }

    def __init__(self, seed=5175):
        try:
            from flybrain import FlyBrain, Trace
        except ImportError as exc:
            raise RuntimeError(
                "flybrain is not installed. Run: pip install -e '.[brain]'"
            ) from exc

        self.FlyBrain = FlyBrain
        self.Trace = Trace
        self.seed = seed

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def semantic_drives(self, vector: SemanticVector) -> dict[str, float]:
        """Map semantic scalars to three documented visual feature channels.

        This encoder is intentionally explicit and inspectable so it can later
        be replaced, trained, randomized, or ablated.
        """
        v = vector.to_dict()

        return {
            # LC4/LPLC2: looming / escape related visual projection neurons.
            "looming_escape": self._clip(
                0.50 * v["threat"]
                + 0.25 * v["avoidance"]
                + 0.15 * v["arousal"]
                + 0.10 * v["motion"]
            ),
            # LPLC1: small approaching objects. We use novelty/motion as a
            # heuristic proxy for a newly appearing moving object.
            "small_approach": self._clip(
                0.40 * v["novelty"]
                + 0.30 * v["motion"]
                + 0.20 * v["uncertainty"]
                + 0.10 * v["arousal"]
            ),
            # LC10a: moving target chased by the male. We use approach/reward
            # only as an experimental semantic-to-sensory encoder.
            "chase_target": self._clip(
                0.45 * v["approach"]
                + 0.25 * v["reward"]
                + 0.20 * v["motion"]
                + 0.10 * v["novelty"]
            ),
        }

    def _command_activity(self, brain, trace, activity: np.ndarray) -> dict[str, float]:
        # trace.idx contains full-brain neuron indices corresponding to activity columns.
        slot = {int(neuron_idx): i for i, neuron_idx in enumerate(trace.idx)}
        outputs: dict[str, float] = {}

        for label, types in self.COMMAND_TYPES.items():
            full_idx = brain.cells(types)
            cols = [slot[int(i)] for i in full_idx if int(i) in slot]
            if not cols:
                outputs[label] = 0.0
                continue

            # Mean over command-neuron columns, then peak over the episode.
            series = activity[:, cols].mean(axis=1)
            outputs[label] = round(float(series.max(initial=0.0)), 6)

        return outputs

    def run(self, vector: SemanticVector, steps=64):
        # sensory_input=False suppresses recurrent input onto sensory neurons,
        # making this a cleaner controlled injection experiment.
        brain = self.FlyBrain(
            device="auto",
            seed=self.seed,
            sensory_input=False,
        )
        brain.reset(self.seed)

        dn_idx = brain.cells(["descending_neuron"])
        if len(dn_idx) == 0:
            raise RuntimeError("No descending_neuron population found in flybrain data")

        trace = self.Trace(brain, idx=dn_idx, tau=0.1)
        drives = self.semantic_drives(vector)

        injection_specs = []
        stimulated: set[int] = set()
        channel_sizes: dict[str, int] = {}

        for channel, cell_types in self.INPUT_TYPES.items():
            idx = brain.cells(cell_types)
            amount = drives[channel]
            channel_sizes[channel] = int(len(idx))
            if len(idx) and amount > 0.0:
                injection_specs.append((idx, amount))
                stimulated.update(int(i) for i in idx)

        # Step manually instead of calling reservoir.run so we can count true
        # network spikes while also collecting the descending-neuron trace.
        snapshots = []
        total_spikes = 0

        for _ in range(int(steps)):
            fired = brain.step(inject=injection_specs)
            total_spikes += len(fired)
            snapshots.append(trace.observe(fired))

        activity = np.stack(snapshots)

        # A DN is active if its trace was ever > 0, which implies at least one
        # spike from that descending neuron during the episode.
        dn_peak = activity.max(axis=0)
        active_neurons = int(np.count_nonzero(dn_peak > 0.0))

        # Integral of the decaying trace is used only for ranking DNs; it is not
        # an exact spike count.
        scores = activity.sum(axis=0)
        order = np.argsort(scores)[::-1][:12]
        top = []
        for col in order:
            full_idx = int(trace.idx[col])
            cell_type = str(brain.cell_type[full_idx])
            side = str(brain.side[full_idx])
            top.append(
                {
                    "neuron_id": full_idx,
                    "cell_type": cell_type,
                    "side": side,
                    "trace_integral": round(float(scores[col]), 6),
                    "peak_trace": round(float(dn_peak[col]), 6),
                }
            )

        output_activity = self._command_activity(brain, trace, activity)

        summary = {
            "network_neurons": int(brain.n),
            "descending_neurons_traced": int(len(trace.idx)),
            "steps": int(steps),
            "dt_seconds": float(brain.dt),
            "simulated_seconds": round(float(steps * brain.dt), 6),
            "semantic_drives": {k: round(float(v), 6) for k, v in drives.items()},
            "input_channel_sizes": channel_sizes,
            "encoder_note": (
                "Semantic-to-sensory mapping is an experimental heuristic. "
                "The connectome wiring is from flybrain/MaleCNS; the text encoder is ours."
            ),
        }

        return FlyResult(
            backend="flybrain",
            stimulated_neurons=sorted(stimulated),
            active_neurons=active_neurons,
            total_spikes=int(total_spikes),
            top_firing_neurons=top,
            output_activity=output_activity,
            neural_activity_summary=summary,
        )


def get_backend(name, seed=5175):
    if name == "proxy":
        return ProxyFlyBrain(seed)
    if name == "flybrain":
        return FlyBrainBackend(seed)
    raise ValueError(name)
