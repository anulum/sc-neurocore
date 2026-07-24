# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_temporal_properties.py

from __future__ import annotations

"""Real-surface tests for temporal spike-train property verification."""
import numpy as np
import numpy.typing as npt
from sc_neurocore.verification.temporal_properties import (
    PropertyResult,
    bounded_activity,
    causal_order,
    fires_within,
    mutual_exclusion,
    rate_bound,
    refractory_guarantee,
)


def _make_spikes(T: int = 50, N: int = 5) -> npt.NDArray[np.int8]:
    """Create an empty binary spike raster with shape ``(T, N)``."""
    return np.zeros((T, N), dtype=np.int8)


__all__ = [
    "np",
    "npt",
    "PropertyResult",
    "bounded_activity",
    "causal_order",
    "fires_within",
    "mutual_exclusion",
    "rate_bound",
    "refractory_guarantee",
    "_make_spikes",
]
