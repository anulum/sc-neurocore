# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reward-modulated STDP synapse (Izhikevich, Cerebral

from __future__ import annotations
from dataclasses import dataclass
import math
from .stochastic_stdp import StochasticSTDPSynapse
from ..constants import RSTDP_TRACE_DECAY, RSTDP_ANTI_HEBBIAN_SCALE


@dataclass
class RewardModulatedSTDPSynapse(StochasticSTDPSynapse):
    """
    Reward-modulated STDP synapse (Izhikevich, Cerebral Cortex 17(10), 2007).

    Eligibility trace accumulates Hebbian coincidences; weight update
    fires only when a global reward signal arrives.

    Example
    -------
    >>> syn = RewardModulatedSTDPSynapse(w_min=0.0, w_max=1.0, w=0.5, length=64)
    >>> for _ in range(20):
    ...     syn.process_step(pre_bit=1, post_bit=1)
    >>> syn.apply_reward(reward=1.0)  # positive reward → potentiate
    >>> syn.w >= 0.5
    True
    """

    eligibility_trace: float = 0.0
    trace_decay: float = RSTDP_TRACE_DECAY
    anti_hebbian_scale: float = RSTDP_ANTI_HEBBIAN_SCALE

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.eligibility_trace):
            raise ValueError("eligibility_trace must be finite")
        if not math.isfinite(self.trace_decay) or not 0.0 <= self.trace_decay <= 1.0:
            raise ValueError("trace_decay must be finite and within [0, 1]")
        if not math.isfinite(self.anti_hebbian_scale) or self.anti_hebbian_scale < 0.0:
            raise ValueError("anti_hebbian_scale must be finite and non-negative")

    def process_step(self, pre_bit: int, post_bit: int) -> int:
        self._validate_bit("pre_bit", pre_bit)
        self._validate_bit("post_bit", post_bit)

        # 1. Compute Output (Same as standard)
        w_prob = self.effective_weight_probability()
        weight_bit = 1 if self._rng.random() < w_prob else 0
        output_bit = pre_bit & weight_bit

        # 2. Update Eligibility Trace instead of Weight
        # (Simplified Hebbian / STDP logic)

        # Hebbian Term: Pre * Post
        # If both fire, trace goes up (Potentiation eligibility)
        if pre_bit == 1 and post_bit == 1:
            self.eligibility_trace += 1.0

        # Anti-Hebbian Term: Pre * !Post (or vice versa depending on rule)
        # If Pre fires but Post doesn't, trace goes down (Depression eligibility)
        elif pre_bit == 1 and post_bit == 0:
            self.eligibility_trace -= self.anti_hebbian_scale

        # Decay trace
        self.eligibility_trace *= self.trace_decay

        return output_bit

    def apply_reward(self, reward: float) -> None:
        """
        Global reward signal triggers weight update.
        """
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")

        # Delta W ~ Reward * Trace
        update = self.learning_rate * reward * self.eligibility_trace

        new_w = self.w + update
        # Clip
        new_w = max(self.w_min, min(self.w_max, new_w))

        self.update_weight(new_w)

        # Optionally reset trace? Usually trace decays naturally.
        # self.eligibility_trace = 0

    @staticmethod
    def _validate_bit(name: str, value: int) -> None:
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"{name} must be an integer bit, 0 or 1")
