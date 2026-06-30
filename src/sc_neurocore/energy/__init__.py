# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pre-silicon energy estimation

"""Pre-silicon energy estimation for FPGA SNN deployment."""

from .estimator import estimate, EnergyReport
from .folded_estimator import estimate_folded_area, FoldedAreaEstimate

__all__ = ["estimate", "EnergyReport", "estimate_folded_area", "FoldedAreaEstimate"]
