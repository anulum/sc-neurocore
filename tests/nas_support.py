# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_nas.py

from __future__ import annotations

import numpy as np
from sc_neurocore.nas.search_space import (
    Architecture,
    SearchSpace,
    NEURON_CHOICES,
    WIDTH_CHOICES,
    L_CHOICES,
)
from sc_neurocore.nas.search import (
    nas,
    NASResult,
    _evaluate,
    _dominates,
    _non_dominated_sort,
    _crowding_distance,
)
from sc_neurocore.nas.equiv import (
    check_equivalence,
    generate_miter,
    generate_sby,
    EquivResult,
)

__all__ = [
    "np",
    "Architecture",
    "SearchSpace",
    "NEURON_CHOICES",
    "WIDTH_CHOICES",
    "L_CHOICES",
    "nas",
    "NASResult",
    "_evaluate",
    "_dominates",
    "_non_dominated_sort",
    "_crowding_distance",
    "check_equivalence",
    "generate_miter",
    "generate_sby",
    "EquivResult",
]
