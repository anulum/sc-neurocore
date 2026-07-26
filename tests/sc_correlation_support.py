# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared stochastic-correlation test support

"""Constructed shared-source streams for stochastic-correlation tests."""

from __future__ import annotations

from typing import Any

import numpy as np


def _shared_source_streams(
    p_a: float, p_b: float, n: int, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Two comonotone streams from one uniform source (SCC == +1)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return (u < p_a).astype(np.uint8), (u < p_b).astype(np.uint8)
