# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gamma oscillation circuit (PING model)

"""Pyramidal-Interneuron Network Gamma (PING) oscillation circuit.

Generates 30-80 Hz gamma oscillations through E/I feedback:
pyramidal cells excite fast-spiking interneurons, which inhibit
the pyramidal cells, creating rhythmic bursting.

Whittington et al., Nature 373:612-615, 1995.
Börgers & Kopell, Neural Computation 15:509-538, 2003.

    from sc_neurocore.network.gamma_oscillation import PINGCircuit

    ping = PINGCircuit(n_excitatory=80, n_inhibitory=20)
    for t in range(1000):
        spikes_e, spikes_i = ping.step(drive=5.0, dt=0.1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PINGCircuit:
    """Pyramidal-Interneuron Network Gamma circuit.

    Parameters
    ----------
    n_excitatory : int
        Number of excitatory (pyramidal) neurons. Default 80.
    n_inhibitory : int
        Number of inhibitory (fast-spiking) neurons. Default 20.
    tau_e : float
        Excitatory membrane time constant (ms). Default 20.
    tau_i : float
        Inhibitory membrane time constant (ms). Default 10 (fast-spiking).
    w_ei : float
        E→I connection weight. Default 0.5.
    w_ie : float
        I→E connection weight (inhibitory). Default 0.8.
    w_ee : float
        E→E recurrent excitation. Default 0.1.
    threshold : float
        Spike threshold. Default 1.0.
    reset : float
        Reset voltage. Default 0.0.
    """

    n_excitatory: int = 80
    n_inhibitory: int = 20
    tau_e: float = 20.0
    tau_i: float = 10.0
    w_ei: float = 0.5
    w_ie: float = 0.8
    w_ee: float = 0.1
    threshold: float = 1.0
    reset: float = 0.0
    v_e: np.ndarray = field(default=None)
    v_i: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.v_e is None:
            self.v_e = np.random.uniform(0, 0.5, self.n_excitatory)
        if self.v_i is None:
            self.v_i = np.random.uniform(0, 0.5, self.n_inhibitory)

    def step(self, drive: float = 5.0, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Advance one timestep.

        Parameters
        ----------
        drive : float
            External drive current to excitatory neurons.
        dt : float
            Timestep in ms.

        Returns
        -------
        (spikes_e, spikes_i): boolean arrays of shape (n_e,) and (n_i,)
        """
        # Compute population firing rates
        rate_e = np.mean(self.v_e > self.threshold * 0.8)
        rate_i = np.mean(self.v_i > self.threshold * 0.8)

        # Excitatory neurons: driven by external input, recurrent E, inhibited by I
        i_e = (
            drive + self.w_ee * rate_e * self.n_excitatory - self.w_ie * rate_i * self.n_inhibitory
        )
        dv_e = (-self.v_e + np.maximum(i_e, 0.0)) * (dt / self.tau_e)
        # Add noise for heterogeneity
        dv_e += np.random.normal(0, 0.05, self.n_excitatory) * np.sqrt(dt)
        self.v_e += dv_e

        # Inhibitory neurons: driven by excitatory population
        i_i = self.w_ei * rate_e * self.n_excitatory
        dv_i = (-self.v_i + np.maximum(i_i, 0.0)) * (dt / self.tau_i)
        dv_i += np.random.normal(0, 0.05, self.n_inhibitory) * np.sqrt(dt)
        self.v_i += dv_i

        # Detect spikes
        spikes_e = self.v_e >= self.threshold
        spikes_i = self.v_i >= self.threshold

        # Reset spiking neurons
        self.v_e[spikes_e] = self.reset
        self.v_i[spikes_i] = self.reset

        return spikes_e, spikes_i

    def reset_state(self):
        self.v_e = np.random.uniform(0, 0.5, self.n_excitatory)
        self.v_i = np.random.uniform(0, 0.5, self.n_inhibitory)
