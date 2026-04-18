# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-spike energy accounting

"""Per-spike, per-synapse, per-layer energy accounting mapped to hardware."""

from .accountant import EnergyAccountant, HardwareCostModel, EnergyReport

__all__ = ["EnergyAccountant", "HardwareCostModel", "EnergyReport"]
