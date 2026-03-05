# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StochasticDendriticNeuron:
    """
    Two-Compartment Neuron (Soma + 2 Dendrites).
    Can solve non-linear problems (XOR) singly.

    Note: This neuron takes *two* inputs (input_a, input_b) rather than
    the single ``input_current`` of :class:`BaseNeuron`.  It therefore
    does not inherit from ``BaseNeuron``, but exposes the same
    ``reset_state()`` / ``get_state()`` interface for consistency.

    Structure:
    Input A -> Dendrite 1
    Input B -> Dendrite 2
    Dendrite Output = NonLinear(Input)
    Soma = Integrate(D1 + D2)
    """

    threshold: float = 1.5
    _last_current: float = field(default=0.0, init=False, repr=False)

    def step(self, input_a: float, input_b: float) -> int:
        """
        Inputs are probabilities/currents.
        """
        d1 = input_a
        d2 = input_b

        # Active Dendrite model with shunting inhibition.
        # Soma Current = D1 + D2 - Interaction(D1*D2)
        current = d1 + d2 - 2.0 * (d1 * d2)
        # Logic:
        # 0,0 -> 0
        # 1,0 -> 1
        # 0,1 -> 1
        # 1,1 -> 1+1 - 2 = 0

        self._last_current = current
        if current > 0.5:
            return 1
        return 0

    def reset_state(self) -> None:
        """Reset internal state to defaults."""
        self._last_current = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Return dict with internal state."""
        return {"last_current": self._last_current, "threshold": self.threshold}
