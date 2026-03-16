# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha-synapse neuron. Rall 1967

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlphaNeuron:
    """Alpha-synapse neuron. Rall 1967.

    Dual excitatory/inhibitory synaptic currents with alpha-function kinetics.
    """

    v: float = 0.0
    i_exc: float = 0.0
    i_inh: float = 0.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    tau_v: float = 20.0
    tau_exc: float = 5.0
    tau_inh: float = 10.0
    dt: float = 1.0

    def step(self, exc_current: float, inh_current: float = 0.0) -> int:
        self.i_exc += (-self.i_exc / self.tau_exc + exc_current) * self.dt
        self.i_inh += (-self.i_inh / self.tau_inh + inh_current) * self.dt
        dv = (-(self.v - self.v_rest) + self.i_exc - self.i_inh) / self.tau_v * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_rest
            return 1
        return 0

    def reset(self):
        self.v = self.v_rest
        self.i_exc = 0.0
        self.i_inh = 0.0
