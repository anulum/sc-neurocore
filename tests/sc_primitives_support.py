# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_primitives.py

"""Shared packed Bernoulli bitstream helpers for the SC primitive gate suites."""

from __future__ import annotations

from typing import Any

import numpy as np
from sc_neurocore.accel.vector_ops import (
    pack_bitstream,
    vec_popcount,
)


def _prob(packed: np.ndarray[Any, Any], length: int) -> float:
    """Estimate probability from packed bitstream."""
    return vec_popcount(packed) / length


def _bernoulli_packed(p: float, length: int, seed: int) -> tuple[np.ndarray[Any, Any], int]:
    """Generate a packed Bernoulli bitstream."""
    rng = np.random.RandomState(seed)
    bits = (rng.random(length) < p).astype(np.uint8)
    return pack_bitstream(bits), length


__all__ = [
    "_prob",
    "_bernoulli_packed",
]
