# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import numpy as np


class KardashevEstimator:
    """
    Calculates Civilization Type on the Kardashev Scale.
    """

    @staticmethod
    def calculate_type(power_watts: float) -> float:
        """
        K = (log10(P) - 6) / 10
        """
        if power_watts <= 0:
            return 0.0
        return (np.log10(power_watts) - 6.0) / 10.0

    @staticmethod
    def estimate_from_compute(ops_per_second: float, efficiency_j_per_op: float = 1e-21) -> float:
        """
        Estimate based on Landauer-limited computing.
        """
        power = ops_per_second * efficiency_j_per_op
        return KardashevEstimator.calculate_type(power)
