# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tripartite synapse: pre-neuron ↔ astrocyte ↔ post-neuron

"""Tripartite synapse with astrocyte-mediated neuromodulation.

The tripartite synapse (Araque et al. 1999, Perea et al. 2009) extends
classical pre-post synaptic transmission with bidirectional astrocyte
coupling:

1. Pre-synaptic spikes release glutamate → drives astrocyte IP3 production
2. Astrocyte Ca²⁺ oscillations → gliotransmitter release (D-serine/ATP)
3. Gliotransmitter modulates synaptic efficacy (facilitation or depression)

This is the first SC+FPGA implementation of tripartite synaptic coupling.

    from sc_neurocore.synapses.tripartite import TripartiteSynapse

    syn = TripartiteSynapse()
    for t in range(10000):
        syn.step(pre_spike=True, post_spike=False, dt=0.01)
"""

from __future__ import annotations

from dataclasses import dataclass

from ..neurons.models.astrocyte import AstrocyteModel


@dataclass
class TripartiteSynapse:
    """Synapse with bidirectional astrocyte coupling.

    Parameters
    ----------
    base_weight : float
        Baseline synaptic weight.
    glut_per_spike : float
        IP3 production rate per pre-synaptic spike (µM/s).
    ca_threshold : float
        Astrocyte Ca²⁺ threshold for gliotransmitter release (µM).
    facilitation : float
        Multiplicative gain when astrocyte is active (> 1 for facilitation).
    depression_rate : float
        Weight depression rate when astrocyte Ca²⁺ is below threshold.
    w_min, w_max : float
        Weight bounds.
    """

    base_weight: float = 0.5
    glut_per_spike: float = 2.0
    ca_threshold: float = 0.3
    facilitation: float = 1.5
    depression_rate: float = 0.001
    w_min: float = 0.0
    w_max: float = 1.0

    def __post_init__(self):
        self.weight = self.base_weight
        self.astrocyte = AstrocyteModel()
        self._glut_current = 0.0  # accumulated glutamate signal

    def step(self, pre_spike: bool, post_spike: bool, dt: float = 0.01) -> float:
        """Advance one timestep.

        Parameters
        ----------
        pre_spike : bool
            Pre-synaptic spike.
        post_spike : bool
            Post-synaptic spike (unused in basic model, reserved for Hebbian extension).
        dt : float
            Timestep in seconds.

        Returns
        -------
        float
            Effective synaptic weight (base_weight * astrocyte modulation).
        """
        # Pre-synaptic activity → glutamate → IP3
        if pre_spike:
            self._glut_current += self.glut_per_spike
        # Glutamate decays
        self._glut_current *= 0.95

        # Step the astrocyte with glutamate-driven IP3 production
        self.astrocyte.dt = dt
        ca = self.astrocyte.step(self._glut_current)

        # Astrocyte modulation of synaptic weight
        if ca > self.ca_threshold:
            # Gliotransmitter release → synaptic facilitation
            self.weight += self.facilitation * (ca - self.ca_threshold) * dt
        else:
            # Slow depression toward baseline without astrocyte support
            self.weight += (self.base_weight - self.weight) * self.depression_rate

        self.weight = max(self.w_min, min(self.w_max, self.weight))
        return self.weight

    @property
    def ca(self) -> float:
        """Current astrocyte Ca²⁺ concentration (µM)."""
        return self.astrocyte.ca

    @property
    def ip3(self) -> float:
        """Current astrocyte IP3 concentration (µM)."""
        return self.astrocyte.ip3

    def effective_weight(self) -> float:
        """Current effective synaptic weight."""
        return self.weight

    def reset(self):
        self.weight = self.base_weight
        self.astrocyte.reset()
        self._glut_current = 0.0
