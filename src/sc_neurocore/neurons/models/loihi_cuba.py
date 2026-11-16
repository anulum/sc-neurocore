# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Loihi CUBA LIF — Intel Loihi fixed-point neuron. Davies 2018

"""Intel Loihi CUBA leaky integrate-and-fire neuron, in integer arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.utils.numerics import trunc_div


@dataclass
class LoihiCUBANeuron:
    """Loihi CUBA LIF — Intel Loihi fixed-point neuron. Davies 2018.

    Each step ``u`` decays by ``u / tau_u`` and gains the input, then ``v`` decays by ``v / tau_v`` and gains ``u``.
    The quotients are truncated towards zero, as the engine and Loihi's sign-magnitude decay round them, so a negative
    state decays as its positive mirror does.

    Reference: Davies, M. et al. (2018). IEEE Micro 38:82–99.
    """

    v: int = 0
    u: int = 0
    tau_v: int = 10
    tau_u: int = 5
    v_threshold: int = 1000
    v_reset: int = 0

    def step(self, weighted_input: int) -> int:
        """Advance one chip step with the integer input; return 1 on a spike, else 0."""
        self.u = self.u - trunc_div(self.u, self.tau_u) + weighted_input
        self.v = self.v - trunc_div(self.v, self.tau_v) + self.u
        if self.v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        return 0

    def reset(self) -> None:
        """Clear both states; the parameters stay."""
        self.v, self.u = 0, 0
