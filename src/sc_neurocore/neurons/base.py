# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Abstract base class for stochastic neuron models

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseNeuron(ABC):
    """
    Abstract base class for stochastic neuron models.

    All neurons should expose:
    - step(input_current) -> spike (0 or 1)
    - reset_state()
    - get_state() -> dict
    """

    @abstractmethod
    def step(self, input_current: float) -> int:
        """Advance the neuron by one time step and return a spike (0 or 1)."""
        raise NotImplementedError

    @abstractmethod
    def reset_state(self) -> None:
        """Reset the internal state to default / initial values."""
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return a dict with the internal state (e.g., membrane potential)."""
        raise NotImplementedError
