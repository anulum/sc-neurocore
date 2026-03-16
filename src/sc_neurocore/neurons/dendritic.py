# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..constants import DENDRITIC_THRESHOLD


@dataclass
class StochasticDendriticNeuron:
    """
    XOR-nonlinearity neuron with shunting inhibition.

    Implements ``d1 + d2 - 2*d1*d2`` (XOR truth table for binary inputs).
    Based on Koch, *Biophysics of Computation*, 1999, Ch. 12.
    """

    threshold: float = DENDRITIC_THRESHOLD
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
