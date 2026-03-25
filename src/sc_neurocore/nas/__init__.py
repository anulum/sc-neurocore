# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware-aware SNN neural architecture search

"""Hardware-aware SNN NAS: search {neuron, width, delays, L} under FPGA budgets."""

from .search_space import Architecture, SearchSpace
from .search import nas, NASResult

__all__ = ["Architecture", "SearchSpace", "nas", "NASResult"]
