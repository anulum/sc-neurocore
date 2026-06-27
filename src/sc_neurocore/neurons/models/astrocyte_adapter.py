# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adapter: AstrocyteModel → Population-compatible neuron

"""Adapter that wraps :class:`AstrocyteModel` for population workflows.

The Li-Rinzel astrocyte model outputs cytosolic calcium concentration rather
than binary spikes. This adapter preserves the model dynamics and exposes the
population-compatible neuron interface used by the network stack:

- ``step(current)`` returns ``1`` when calcium exceeds ``ca_threshold``.
- ``v`` reports calcium concentration as a pseudo-voltage.
- ``reset()`` delegates to ``AstrocyteModel.reset()`` and refreshes ``v``.

    from sc_neurocore.neurons.models.astrocyte_adapter import AstrocyteNeuron
    from sc_neurocore.network.population import Population

    pop = Population(AstrocyteNeuron, n=10, params={"ca_threshold": 0.3})

Reference: De Pittà, M. et al. (2011). J. Biol. Phys. 37:195–230.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .astrocyte import AstrocyteModel


@dataclass
class AstrocyteNeuron:
    """Population-compatible wrapper for AstrocyteModel.

    Parameters
    ----------
    ca_threshold : float
        Ca²⁺ concentration (µM) above which the astrocyte "fires"
        (releases gliotransmitter). Default 0.3 µM.
    dt : float
        Timestep in seconds.
    """

    ca_threshold: float = 0.3
    dt: float = 0.01

    def __post_init__(self) -> None:
        """Initialise the wrapped astrocyte model and exposed pseudo-voltage."""
        if not math.isfinite(self.ca_threshold) or self.ca_threshold < 0.0:
            raise ValueError("ca_threshold must be finite and non-negative")
        self._astro = AstrocyteModel(dt=self.dt)
        self.v = self._astro.ca

    def step(self, current: float) -> int:
        """Advance one timestep and return the thresholded release event.

        Parameters
        ----------
        current : float
            Glutamate-driven IP3 production drive passed through to the wrapped
            astrocyte model.

        Returns
        -------
        int
            ``1`` when cytosolic calcium exceeds ``ca_threshold``; otherwise
            ``0``.
        """
        ca = self._astro.step(current)
        self.v = ca
        return 1 if ca > self.ca_threshold else 0

    @property
    def ca(self) -> float:
        """Current cytosolic calcium concentration in micromolar."""
        return self._astro.ca

    @property
    def ip3(self) -> float:
        """Current IP3 concentration in micromolar."""
        return self._astro.ip3

    def reset(self) -> None:
        """Reset the wrapped astrocyte state and pseudo-voltage."""
        self._astro.reset()
        self.v = self._astro.ca
