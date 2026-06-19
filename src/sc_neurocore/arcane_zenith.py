# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ArcaneZenith Cognitive Core

"""ArcaneZenith Cognitive Core.

Wires the ArcaneNeuron self-modeling continuous architecture directly to the Zenith
plasticity hardware ecosystem. The neuron's own meta-parameters (tau, thresholds, learning rates)
are fully controlled by dynamically adapting synaptic plasticity traces driven by structural novelty.
"""

from typing import Dict, Any, Sequence

import numpy as np

from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron
from sc_neurocore.plasticity import create_plasticity_layer
from sc_neurocore.evo_substrate.evo_substrate import Genome
from sc_neurocore.fault_injection import (
    FaultInjectionResilienceMode,
    RadiationProfile,
    ResilienceModeConfig,
)


class ArcaneZenithCognitiveCore:
    """A self-improving cognitive primitive combining ArcaneNeuron and Zenith plasticity.

    Rather than maintaining static deep-context parameters, the ArcaneZenith module
    deploys 4 synchronized Zenith meta-plasticity connections controlling physical limits.
    Zenith plasticity weights ∈ [0, 1] are smoothly mapped to safe biological ranges
    for each parameter using a sigmoid interpolator.

    Example:
        >>> core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        >>> for i in range(10):
        ...     spike = core.step(current=i % 50)
        >>> print(f"drift={core.neuron.identity_drift:.4f}")
    """

    def __init__(self, backend: str = "torch", **kwargs: Any) -> None:
        self.neuron = ArcaneNeuron()
        self._reasoning_tick = 0
        self._last_reasoning_state: dict[str, float] | None = None

        # RULE_REWARD_STDP seamlessly interpolates weights bounded [0, 1] mapped dynamically to limits
        self.tau_rule = create_plasticity_layer(
            count=1, rule_type=2, backend=backend, weight=0.5, **kwargs
        )
        self.nov_rule = create_plasticity_layer(
            count=1, rule_type=2, backend=backend, weight=0.2, **kwargs
        )
        self.conf_rule = create_plasticity_layer(
            count=1, rule_type=2, backend=backend, weight=0.3, **kwargs
        )
        self.lr_rule = create_plasticity_layer(
            count=1, rule_type=2, backend=backend, weight=0.1, **kwargs
        )

    def _map_to_range(self, w: float, min_val: float, max_val: float) -> float:
        """Smooth sigmoid mapping centered at 0.5 to prevent edge explosions."""
        t = float(1.0 / (1.0 + np.exp(-10.0 * (w - 0.5))))
        return max(min_val, min(max_val, min_val + t * (max_val - min_val)))

    def step(self, current: float) -> int:
        """Step the unified physical simulation one tick forward."""

        # Pre-spike trace boundary: use the neuron's short-history as pre-synaptic activation proxy
        pre_proxy = np.array([bool(self.neuron.get_recent_pre_activity())], dtype=bool)

        spike = self.neuron.step(current)
        post_proxy = np.array([bool(spike)], dtype=bool)

        # The novelty scalar actively bridges cognitive identity drift back to Zenith structural traces
        reward = np.array([self.neuron.novelty], dtype=np.float32)

        self.tau_rule.step(pre_proxy, post_proxy, reward)
        self.nov_rule.step(pre_proxy, post_proxy, reward)
        self.conf_rule.step(pre_proxy, post_proxy, reward)
        self.lr_rule.step(pre_proxy, post_proxy, reward)

        # Recover structural bounds and map phenomenological scales interpolation
        w_tau = float(self.tau_rule.get_weights()[0])
        w_nov = float(self.nov_rule.get_weights()[0])
        w_conf = float(self.conf_rule.get_weights()[0])
        w_lr = float(self.lr_rule.get_weights()[0])

        # Interpolate meta-parameters smoothly bounded across verifiable biological domains
        self.neuron.tau_deep = self._map_to_range(w_tau, 1000.0, 50000.0)
        self.neuron.surprise_baseline = self._map_to_range(w_nov, 0.01, 0.5)
        self.neuron.delta_conf = self._map_to_range(w_conf, 0.0, 1.0)
        self.neuron.lr_base = self._map_to_range(w_lr, 0.001, 0.1)
        self._reasoning_tick += 1

        return spike

    @staticmethod
    def _level(value: float, *, low: float, high: float) -> str:
        if value < low:
            return "low"
        if value > high:
            return "high"
        return "medium"

    @staticmethod
    def _trend(delta: float, *, eps: float = 1e-6) -> str:
        if delta > eps:
            return "rising"
        if delta < -eps:
            return "falling"
        return "steady"

    def step_from_bio_rates(self, rates: Dict[int, float]) -> None:
        """Modulate phenomenological bounds leveraging a multi-channel biological firing rate map.

        Evaluates the aggregate biological rate to drive structural novelty and parameter progression.
        """
        # Mean population rate determines structural excitation
        mean_rate = np.mean(list(rates.values())) if rates else 0.0

        # Step the unified physical simulation one tick forward mapped to the mean bio rate
        self.step(float(mean_rate))

    def evaluate_bio_pathway_resilience(
        self,
        rates: Dict[int, float],
        *,
        pathway_name: str,
        bitstream_length: int = 256,
        radiation_profile: RadiationProfile | None = None,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Run deterministic fault-injection resilience over biological pathways.

        Converts each pathway-rate channel into a reproducible stochastic bitstream and
        evaluates it through the seeded resilience mode.
        """
        if not pathway_name:
            raise ValueError("pathway_name must be non-empty")
        if bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")

        profile = radiation_profile or RadiationProfile(
            "zenith-bio-default",
            1e-4,
            "Default biological pathway resilience stress",
        )
        bitstreams = self._pathway_bitstreams(rates, bitstream_length=bitstream_length, seed=seed)
        config = ResilienceModeConfig(
            layer_id=f"bio:{pathway_name}",
            radiation_profile=profile,
            seed=seed,
        )
        report = FaultInjectionResilienceMode(config).run(bitstreams)
        payload = report.to_dict()
        payload["pathway_name"] = pathway_name
        payload["pathway_channels"] = sorted(int(key) for key in rates)
        return payload

    @staticmethod
    def _pathway_bitstreams(
        rates: Dict[int, float],
        *,
        bitstream_length: int,
        seed: int,
    ) -> np.ndarray[Any, Any]:
        channels = sorted(int(key) for key in rates)
        if not channels:
            return np.zeros((1, bitstream_length), dtype=np.uint8)

        values = np.array(
            [max(0.0, float(rates[channel])) for channel in channels], dtype=np.float64
        )
        max_rate = float(np.max(values))
        if max_rate <= 0.0:
            probs = np.zeros_like(values)
        else:
            probs = np.clip(values / max_rate, 0.0, 1.0)

        rng = np.random.default_rng(seed)
        draws = rng.random((len(channels), bitstream_length))
        return (draws < probs[:, None]).astype(np.uint8)

    def run_meta_learning_episode(
        self,
        currents: Sequence[float],
        *,
        reset_before: bool = False,
    ) -> Dict[str, Any]:
        """Run a full outer-loop adaptation episode over a current sequence.

        This is the production-facing built-in meta-learning loop for ArcaneZenith:
        each input current advances neuron state, updates four Zenith plasticity
        rules, remaps bounded meta-parameters, and records a deterministic trace.
        """
        if reset_before:
            self.reset()
        if not currents:
            raise ValueError("currents must be non-empty")

        trace: list[dict[str, Any]] = []
        spikes = 0

        for current in currents:
            spike = int(self.step(float(current)))
            spikes += spike
            state = self.get_state()
            trace.append(
                {
                    "current": float(current),
                    "spike": spike,
                    "tau_deep": float(self.neuron.tau_deep),
                    "surprise_baseline": float(self.neuron.surprise_baseline),
                    "delta_conf": float(self.neuron.delta_conf),
                    "lr_base": float(self.neuron.lr_base),
                    "novelty": float(state["novelty"]),
                    "confidence": float(state["confidence"]),
                    "identity_drift": float(state["identity_drift"]),
                    "symbolic_log": self.export_symbolic_reasoning_log(),
                }
            )

        return {
            "steps": len(currents),
            "spike_count": spikes,
            "spike_rate": float(spikes / len(currents)),
            "final_state": self.get_state(),
            "trace": trace,
        }

    def export_reasoning_trace(self) -> Dict[str, float]:
        """Export a compact symbolic trace for outer-loop introspection."""
        state = self.get_state()
        return {
            "novelty": float(state["novelty"]),
            "confidence": float(state["confidence"]),
            "identity_drift": float(state["identity_drift"]),
            "tau_deep": float(self.neuron.tau_deep),
            "surprise_baseline": float(self.neuron.surprise_baseline),
            "delta_conf": float(self.neuron.delta_conf),
            "lr_base": float(self.neuron.lr_base),
        }

    def export_symbolic_reasoning_log(self) -> Dict[str, Any]:
        """Export a short symbolic self-verification log for downstream audit."""
        current = self.export_reasoning_trace()
        previous = self._last_reasoning_state

        novelty = current["novelty"]
        confidence = current["confidence"]
        drift = current["identity_drift"]
        if previous is None:
            confidence_delta = 0.0
            drift_delta = 0.0
            novelty_delta = 0.0
        else:
            confidence_delta = confidence - previous["confidence"]
            drift_delta = drift - previous["identity_drift"]
            novelty_delta = novelty - previous["novelty"]

        self._last_reasoning_state = dict(current)

        return {
            "schema_version": "sc-neurocore.arcane-zenith.symbolic-reasoning-log.v1",
            "tick": self._reasoning_tick,
            "novelty_level": self._level(novelty, low=0.33, high=0.66),
            "novelty_shift": self._trend(novelty_delta, eps=1e-3),
            "confidence_trend": self._trend(confidence_delta, eps=1e-4),
            "identity_shift": "drifting" if drift_delta > 1e-6 else "stable",
            "adaptation_regime": ("aggressive" if current["lr_base"] > 0.05 else "conservative"),
            "evidence": {
                "novelty": novelty,
                "confidence": confidence,
                "identity_drift": drift,
                "tau_deep": current["tau_deep"],
                "delta_conf": current["delta_conf"],
                "lr_base": current["lr_base"],
            },
        }

    def step_from_genome(self, genome: Genome) -> None:
        """Modulate phenomenological bounds leveraging a generated Evo Substrate Genome.

        Evaluates the organism's parameters to drive structural novelty and progression.
        """
        self.neuron.tau_deep = genome.neuron.tau_deep
        self.neuron.tau_fast = genome.neuron.tau_fast
        self.neuron.tau_work = genome.neuron.tau_work
        self.step(float(genome.topology.connectivity))

    def reset(self) -> None:
        self.neuron.reset()
        self.tau_rule.reset()
        self.nov_rule.reset()
        self.conf_rule.reset()
        self.lr_rule.reset()

    def get_state(self) -> Dict[str, Any]:
        """Output serialized limits combining Arcane and Zenith structures natively."""
        state = self.neuron.get_state()
        state.update(
            {
                "w_tau": float(self.tau_rule.get_weights()[0]),
                "w_nov": float(self.nov_rule.get_weights()[0]),
                "w_conf": float(self.conf_rule.get_weights()[0]),
                "w_lr": float(self.lr_rule.get_weights()[0]),
            }
        )
        return state

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "tau_rule": self.tau_rule.get_state_dict(),
            "nov_rule": self.nov_rule.get_state_dict(),
            "conf_rule": self.conf_rule.get_state_dict(),
            "lr_rule": self.lr_rule.get_state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.tau_rule.load_state_dict(state_dict["tau_rule"])
        self.nov_rule.load_state_dict(state_dict["nov_rule"])
        self.conf_rule.load_state_dict(state_dict["conf_rule"])
        self.lr_rule.load_state_dict(state_dict["lr_rule"])


def create_arcane_neuron_with_zenith_plasticity(
    backend: str = "torch", **kwargs: Any
) -> ArcaneZenithCognitiveCore:
    """Seamless factory configuring a unified ArcaneZenith primitive running entirely connected."""
    return ArcaneZenithCognitiveCore(backend=backend, **kwargs)
