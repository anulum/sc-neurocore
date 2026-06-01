# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Oster, Douglas & Liu 2009 — winner-take-all with lateral

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite


@dataclass
class LeakyCompeteFireNeuron:
    """Oster, Douglas & Liu 2009 — winner-take-all with lateral inhibition.

    Reference: Oster, M. et al. (2009). Neural Comput. 21(9):2437–2465.
    """

    n_units: int = 4
    v: list[float] = field(default_factory=lambda: [0.0] * 4)
    tau: float = 10.0
    v_threshold: float = 1.0
    w_inh: float = 0.5
    dt: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.n_units, int) or self.n_units <= 0:
            raise ValueError("n_units must be positive")
        self._validate_parameters()
        if len(self.v) != self.n_units:
            if self.v == [0.0] * 4:
                self.v = [0.0] * self.n_units
            else:
                raise ValueError(f"v must have length {self.n_units}")
        if any(not isfinite(voltage) for voltage in self.v):
            raise ValueError("v must contain only finite values")

    def _validate_parameters(self) -> None:
        if not isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if not isfinite(self.v_threshold):
            raise ValueError("v_threshold must be finite")
        if not isfinite(self.w_inh) or self.w_inh < 0.0:
            raise ValueError("w_inh must be finite and non-negative")
        if not isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

    def _validate_runtime_state(self) -> None:
        self._validate_parameters()
        if len(self.v) != self.n_units:
            raise ValueError(f"v must have length {self.n_units}")
        if any(not isfinite(voltage) for voltage in self.v):
            raise ValueError("v must contain only finite values")

    def _normalise_currents(self, currents: list[float] | float) -> list[float]:
        if isinstance(currents, (int, float)):
            current_values = [float(currents)] * self.n_units
        else:
            current_values = [float(current) for current in currents]
        if len(current_values) != self.n_units:
            raise ValueError(f"currents must have length {self.n_units}")
        if any(not isfinite(current) for current in current_values):
            raise ValueError("currents must contain only finite values")
        return current_values

    def step(self, currents: list[float] | float) -> list[int]:
        current_values = self._normalise_currents(currents)
        self._validate_runtime_state()
        decay = exp(-self.dt / self.tau)
        next_v = [
            current + (voltage - current) * decay
            for voltage, current in zip(self.v, current_values)
        ]
        if any(not isfinite(voltage) for voltage in next_v):
            raise ValueError("LCF exact relaxation produced a non-finite candidate")
        spikes = [0] * self.n_units
        for i in range(self.n_units):
            if next_v[i] >= self.v_threshold:
                spikes[i] = 1
                next_v[i] = 0.0
                for j in range(self.n_units):
                    if j != i:
                        next_v[j] = max(0.0, next_v[j] - self.w_inh)
        self.v = next_v
        return spikes

    def reset(self) -> None:
        self.v = [0.0] * self.n_units
