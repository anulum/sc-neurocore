# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from dataclasses import dataclass
from .stochastic_stdp import StochasticSTDPSynapse


@dataclass
class RewardModulatedSTDPSynapse(StochasticSTDPSynapse):
    """
    Reward-Modulated STDP Synapse.

    Instead of updating weights immediately, we update an 'eligibility trace'.
    Weights are only updated when a 'reward' signal is applied.

    Delta_W = Learning_Rate * Reward * Eligibility
    """

    eligibility_trace: float = 0.0
    trace_decay: float = 0.9
    anti_hebbian_scale: float = 0.5  # LTD/LTP asymmetry; Bi & Poo 1998

    def process_step(self, pre_bit: int, post_bit: int) -> int:
        # 1. Compute Output (Same as standard)
        w_prob = self.effective_weight_probability()
        weight_bit = 1 if self._rng.random() < w_prob else 0  # type: ignore
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

    def apply_reward(self, reward: float):  # type: ignore
        """
        Global reward signal triggers weight update.
        """
        # Delta W ~ Reward * Trace
        update = self.learning_rate * reward * self.eligibility_trace

        new_w = self.w + update
        # Clip
        new_w = max(self.w_min, min(self.w_max, new_w))

        self.update_weight(new_w)

        # Optionally reset trace? Usually trace decays naturally.
        # self.eligibility_trace = 0
