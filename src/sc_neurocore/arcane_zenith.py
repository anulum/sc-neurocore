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

from typing import Dict, Any

import numpy as np

from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron
from sc_neurocore.plasticity import create_plasticity_layer
from sc_neurocore.evo_substrate.evo_substrate import Genome


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

    def __init__(self, backend: str = "torch", **kwargs) -> None:
        self.neuron = ArcaneNeuron()

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
        t = 1.0 / (1.0 + np.exp(-10.0 * (w - 0.5)))
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

        return spike

    def step_from_bio_rates(self, rates: Dict[int, float]) -> None:
        """Modulate phenomenological bounds leveraging a multi-channel biological firing rate map.

        Evaluates the aggregate biological rate to drive structural novelty and parameter progression.
        """
        # Mean population rate determines structural excitation
        mean_rate = np.mean(list(rates.values())) if rates else 0.0

        # Step the unified physical simulation one tick forward mapped to the mean bio rate
        self.step(float(mean_rate))

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
    backend: str = "torch", **kwargs
) -> ArcaneZenithCognitiveCore:
    """Seamless factory configuring a unified ArcaneZenith primitive running entirely connected."""
    return ArcaneZenithCognitiveCore(backend=backend, **kwargs)
