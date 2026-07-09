# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import numpy as np
from dataclasses import dataclass


@dataclass
class DysonPowerGrid:
    """
    Manages energy distribution for a Dyson Swarm.
    """

    n_collectors: int
    n_consumers: int

    def __post_init__(self):
        # Efficiency of each collector [0, 1]
        self.collector_efficiency = np.random.uniform(0.8, 1.0, self.n_collectors)
        # Power demand of consumers
        self.consumer_demand = np.random.uniform(1.0, 10.0, self.n_consumers)

    def step(self, solar_output: float) -> float:
        """
        Simulate one time step.
        solar_output: Total star output (Watts).
        """
        # 1. Harvest
        # Simulate occlusion/failures
        active_collectors = np.random.random(self.n_collectors) > 0.01  # 99% uptime
        total_harvest = np.sum(self.collector_efficiency[active_collectors]) * (
            solar_output / self.n_collectors
        )

        # 2. Distribute
        total_demand = np.sum(self.consumer_demand)
        satisfaction = min(1.0, total_harvest / (total_demand + 1e-9))

        return satisfaction
