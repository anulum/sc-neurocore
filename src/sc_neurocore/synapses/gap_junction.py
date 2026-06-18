# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gap junction (electrical synapse)

"""Gap junction: bidirectional conductance-coupled electrical synapse.

I_gap = g_c * (V_pre - V_post)

Unlike chemical synapses (unidirectional, spike-triggered), gap junctions
pass current proportional to the voltage difference between two neurons.
They enable subthreshold signal sharing, fast synchronization, and
network-wide oscillatory coupling.

    from sc_neurocore.synapses.gap_junction import GapJunction

    gj = GapJunction(conductance=0.1)
    i_to_post = gj.current(v_pre=-50.0, v_post=-65.0)  # = 1.5
    i_to_pre = gj.current(v_pre=-65.0, v_post=-50.0)   # = -1.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np


@dataclass
class GapJunction:
    """Bidirectional electrical synapse.

    Parameters
    ----------
    conductance : float
        Gap junction conductance g_c (nS). Typical: 0.01-1.0 nS.
        Bennett & Zukin, Neuron 2004.
    rectification : float
        Rectification factor in [0, 1]. 0 = fully bidirectional (ohmic),
        1 = fully rectifying (current flows in one direction only).
        Default 0 (standard gap junction).
    """

    conductance: float = 0.1
    rectification: float = 0.0

    def __post_init__(self) -> None:
        """Validate the conductance and rectification parameters."""
        if not math.isfinite(self.conductance) or self.conductance < 0.0:
            raise ValueError("conductance must be finite and non-negative")
        if not math.isfinite(self.rectification) or not 0.0 <= self.rectification <= 1.0:
            raise ValueError("rectification must be finite and within [0, 1]")

    def current(self, v_pre: float, v_post: float) -> float:
        """Compute gap junction current flowing INTO v_post.

        I_gap = g_c * (V_pre - V_post) * rectification_factor

        Positive current depolarizes post. The same junction produces
        equal and opposite current for the pre-synaptic neuron.
        """
        if not math.isfinite(v_pre) or not math.isfinite(v_post):
            raise ValueError("voltage inputs must be finite")

        dv = v_pre - v_post
        if self.rectification > 0:
            # Rectification: reduce current in one direction
            factor = 1.0 - self.rectification * (1.0 if dv < 0 else 0.0)
            return self.conductance * dv * factor
        return self.conductance * dv

    def current_matrix(
        self, voltages: np.ndarray[Any, Any], adjacency: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Compute gap junction currents for a population.

        Parameters
        ----------
        voltages : np.ndarray, shape (N,)
            Membrane voltages of all neurons.
        adjacency : np.ndarray, shape (N, N)
            Binary or weighted adjacency matrix. A[i,j] = 1 means
            neurons i and j are connected by a gap junction.

        Returns
        -------
        np.ndarray, shape (N,)
            Net gap junction current for each neuron.
        """
        self._validate_current_matrix_inputs(voltages, adjacency)

        N = len(voltages)
        dv_matrix = voltages[np.newaxis, :] - voltages[:, np.newaxis]  # dv[i,j] = V[j] - V[i]
        currents = self.conductance * dv_matrix * adjacency
        net_current: np.ndarray[Any, Any] = currents.sum(axis=1)
        return net_current

    @staticmethod
    def _validate_current_matrix_inputs(
        voltages: np.ndarray[Any, Any], adjacency: np.ndarray[Any, Any]
    ) -> None:
        if not isinstance(voltages, np.ndarray) or voltages.ndim != 1:
            raise ValueError("current_matrix voltages must be a one-dimensional array")
        if not np.all(np.isfinite(voltages)):
            raise ValueError("current_matrix voltages must be finite")
        if not isinstance(adjacency, np.ndarray) or adjacency.ndim != 2:
            raise ValueError("current_matrix adjacency must be a two-dimensional array")
        if adjacency.shape != (voltages.shape[0], voltages.shape[0]):
            raise ValueError("current_matrix adjacency must be square with one row per voltage")
        if not np.all(np.isfinite(adjacency)):
            raise ValueError("current_matrix adjacency must be finite")
        if np.any(adjacency < 0.0):
            raise ValueError("current_matrix adjacency must be non-negative")
        if not np.allclose(adjacency, adjacency.T):
            raise ValueError("current_matrix adjacency must be symmetric for reciprocal coupling")
