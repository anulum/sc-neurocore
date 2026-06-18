# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rall branching model for dendritic computation

"""Dendritic tree with Rall's 3/2 power rule for impedance matching.

Rall 1959: at a branch point, the parent dendrite diameter d_p relates
to daughter diameters d_1, d_2 by: d_p^(3/2) = d_1^(3/2) + d_2^(3/2).

This ensures impedance matching and uniform electrotonic propagation.
The tree topology enables nonlinear dendritic computation — synaptic
inputs on the same branch interact multiplicatively, while inputs on
different branches sum linearly at the soma.

    from sc_neurocore.layers.rall_dendrite import RallDendrite

    dendrite = RallDendrite(n_branches=4, branch_length=3)
    soma_v = dendrite.step(branch_inputs)
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any
import numpy as np


@dataclass
class RallDendrite:
    """Dendritic tree with Rall branching and compartmental dynamics.

    Parameters
    ----------
    n_branches : int
        Number of dendritic branches.
    branch_length : int
        Number of compartments per branch.
    tau : float
        Membrane time constant (ms).
    coupling : float
        Inter-compartment coupling strength (0 to 1).
    dt : float
        Timestep (ms).
    """

    n_branches: int = 4
    branch_length: int = 3
    tau: float = 10.0
    coupling: float = 0.5
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate the dendrite geometry and initialise compartment state."""
        if self.n_branches <= 0:
            raise ValueError("n_branches must be positive")
        if self.branch_length <= 0:
            raise ValueError("branch_length must be positive")
        if not np.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.coupling) or not (0.0 <= self.coupling <= 1.0):
            raise ValueError("coupling must be finite and in [0, 1]")

        # Each branch has branch_length compartments
        # Compartment voltages: shape (n_branches, branch_length)
        self.v = np.zeros((self.n_branches, self.branch_length))
        self.soma_v = 0.0
        self._decay = np.exp(-self.dt / self.tau)
        # Rall 3/2 rule: branch diameters for impedance matching
        # Daughter diameters normalised so d_parent^1.5 = sum(d_i^1.5)
        self.diameters = np.ones(self.n_branches)
        parent_d = (self.n_branches) ** (2.0 / 3.0)
        self.attenuation = (self.diameters / parent_d) ** 1.5

    def step(self, branch_inputs: np.ndarray[Any, Any]) -> float:
        """Advance one timestep.

        Parameters
        ----------
        branch_inputs : np.ndarray[Any, Any]
            Shape (n_branches,) — synaptic current injected at distal tip of each branch.

        Returns
        -------
        float
            Somatic voltage.
        """
        branch_inputs = np.asarray(branch_inputs, dtype=np.float64)
        if branch_inputs.shape != (self.n_branches,):
            raise ValueError(
                f"branch_inputs must have shape ({self.n_branches},), got {branch_inputs.shape}"
            )
        if not np.all(np.isfinite(branch_inputs)):
            raise ValueError("branch_inputs must contain only finite values")

        # Decay all compartments
        self.v *= self._decay

        # Inject input at distal tip (last compartment)
        self.v[:, -1] += branch_inputs * self.dt / self.tau

        # Propagate along branch: distal → proximal (toward soma)
        for k in range(self.branch_length - 1, 0, -1):
            flow = self.coupling * (self.v[:, k] - self.v[:, k - 1])
            self.v[:, k] -= flow
            self.v[:, k - 1] += flow

        # Sum proximal compartments at soma with Rall attenuation
        proximal = self.v[:, 0]
        soma_input = np.sum(proximal * self.attenuation)
        self.soma_v = self._decay * self.soma_v + soma_input * self.dt / self.tau

        return float(self.soma_v)

    @property
    def branch_voltages(self) -> np.ndarray[Any, Any]:
        """Current compartment voltages, shape (n_branches, branch_length)."""
        return self.v.copy()

    def reset(self) -> None:
        """Reset all compartment and soma voltages to zero."""
        self.v[:] = 0.0
        self.soma_v = 0.0
