# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel-Wang LIF with NMDA/AMPA/GABA synapses

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BrunelWangNeuron:
    """LIF neuron with NMDA, AMPA, and GABA synaptic currents.

    Brunel, N. & Wang, X.J. (2001). Effects of neuromodulation in a
    cortical network model of object working memory dominated by
    recurrent inhibition. J Comput Neurosci 11:63-85.

    Used in decision-making and working memory models. The key feature
    is the voltage-dependent NMDA conductance with Mg2+ block.
    """

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -55.0
    v_threshold: float = -50.0
    tau_m: float = 20.0
    tau_ref: float = 2.0
    tau_ampa: float = 2.0
    tau_nmda_rise: float = 2.0
    tau_nmda_decay: float = 100.0
    tau_gaba: float = 5.0
    g_ampa_ext: float = 2.1
    g_ampa_rec: float = 0.05
    g_nmda: float = 0.165
    g_gaba: float = 1.3
    v_ampa: float = 0.0
    v_nmda: float = 0.0
    v_gaba: float = -70.0
    C_m: float = 0.5
    mg_conc: float = 1.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        self._s_ampa = 0.0
        self._s_nmda = 0.0
        self._x_nmda = 0.0
        self._s_gaba = 0.0
        self._ref_remaining = 0.0

    def _nmda_voltage_dep(self, v: float) -> float:
        """Mg2+ block factor: 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))."""
        return 1.0 / (1.0 + self.mg_conc / 3.57 * np.exp(-0.062 * v))

    def step(
        self,
        i_ampa_ext: float = 0.0,
        s_ampa_rec: float = 0.0,
        s_nmda_rec: float = 0.0,
        s_gaba: float = 0.0,
    ) -> int:
        """Advance one timestep.

        Parameters
        ----------
        i_ampa_ext : external AMPA current (from Poisson input)
        s_ampa_rec : recurrent AMPA synaptic variable [0, 1]
        s_nmda_rec : recurrent NMDA synaptic variable [0, 1]
        s_gaba : inhibitory GABA synaptic variable [0, 1]
        """
        if self._ref_remaining > 0:
            self._ref_remaining -= self.dt
            return 0

        # Synaptic currents
        i_ampa = -self.g_ampa_ext * (self.v - self.v_ampa) * i_ampa_ext
        i_ampa += -self.g_ampa_rec * (self.v - self.v_ampa) * s_ampa_rec
        i_nmda = -self.g_nmda * self._nmda_voltage_dep(self.v) * (self.v - self.v_nmda) * s_nmda_rec
        i_gaba = -self.g_gaba * (self.v - self.v_gaba) * s_gaba

        # Membrane dynamics
        i_leak = -(self.v - self.v_rest) / self.tau_m
        dv = (i_leak + (i_ampa + i_nmda + i_gaba) / self.C_m) * self.dt
        self.v += dv

        if self.v >= self.v_threshold:
            self.v = self.v_reset
            self._ref_remaining = self.tau_ref
            return 1
        return 0

    def reset(self) -> None:
        self.v = self.v_rest
        self._s_ampa = 0.0
        self._s_nmda = 0.0
        self._x_nmda = 0.0
        self._s_gaba = 0.0
        self._ref_remaining = 0.0

    def get_state(self):
        return {"v": self.v, "ref_remaining": self._ref_remaining}
