# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class PlanetarySensorGrid:
    """
    Planetary-Scale Computing (Gaia Interface).
    Aggregates global telemetry into an SC computational field.
    """

    n_nodes: int = 1000000  # Default 1M nodes

    def aggregate_field(self, telemetry_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuses multi-regional telemetry (e.g. Temperature, CO2, Humidity)
        into a global probability field for the SC processor.
        """
        field = np.zeros(self.n_nodes)
        count = 0
        for key, data in telemetry_data.items():
            # Assume data maps to regions or nodes
            # We normalize and accumulate
            norm_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
            field[: len(norm_data)] += norm_data
            count += 1

        return field / count if count > 0 else field
