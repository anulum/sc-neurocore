# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adapter: AstrocyteModel → Population-compatible neuron

"""Adapter that wraps AstrocyteModel for use in Population/Network.

The AstrocyteModel (Li-Rinzel 1994) outputs Ca²⁺ concentration,
not spikes. This adapter converts it to the Population neuron interface:

- step(current) → int (0 or 1): spike when Ca > threshold
- v attribute: reports Ca concentration as pseudo-voltage
- reset(): delegates to AstrocyteModel.reset()

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
        if not math.isfinite(self.ca_threshold) or self.ca_threshold < 0.0:
            raise ValueError("ca_threshold must be finite and non-negative")
        self._astro = AstrocyteModel(dt=self.dt)
        self.v = self._astro.ca

    def step(self, current: float) -> int:
        """Advance one timestep. Returns 1 if Ca > threshold, else 0."""
        ca = self._astro.step(current)
        self.v = ca
        return 1 if ca > self.ca_threshold else 0

    @property
    def ca(self) -> float:
        return self._astro.ca

    @property
    def ip3(self) -> float:
        return self._astro.ip3

    def reset(self) -> None:
        self._astro.reset()
        self.v = self._astro.ca
