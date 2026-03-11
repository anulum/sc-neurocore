# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StochasticDendriticNeuron:
    """
    XOR-nonlinearity neuron with shunting inhibition.

    Implements ``d1 + d2 - 2*d1*d2`` which gives XOR truth table
    for binary inputs. Not a Rall cable-equation compartmental model.

    Takes two inputs (input_a, input_b). Does not inherit BaseNeuron
    because ``step()`` has a 2-input signature incompatible with the ABC.
    """

    threshold: float = 0.5
    _last_current: float = field(default=0.0, init=False, repr=False)

    def step(self, input_a: float, input_b: float) -> int:
        d1 = input_a
        d2 = input_b

        # XOR nonlinearity: d1 + d2 - 2*d1*d2
        current = d1 + d2 - 2.0 * (d1 * d2)

        self._last_current = current
        if current > self.threshold:
            return 1
        return 0

    def reset_state(self) -> None:
        """Reset internal state to defaults."""
        self._last_current = 0.0

    def get_state(self) -> Dict[str, Any]:
        """Return dict with internal state."""
        return {"last_current": self._last_current, "threshold": self.threshold}
