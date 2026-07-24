# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_topology_generators.py

from __future__ import annotations

"""Tests for all 6 connectivity generators returning CSR arrays."""
from typing import Any, cast
import numpy as np
import pytest
from sc_neurocore.network.topology import (
    random_connectivity,
    small_world,
    scale_free,
    ring_topology,
    grid_topology,
    all_to_all,
)


def _csr_to_dense(
    indptr: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    n_rows: int,
    n_cols: int,
) -> np.ndarray[Any, Any]:
    """Convert a CSR tuple to a dense matrix for structural assertions."""
    mat = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for k in range(indptr[i], indptr[i + 1]):
            mat[i, indices[k]] = data[k]
    return mat


def _validate_csr(
    indptr: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    n_rows: int,
    n_cols: int,
) -> None:
    """Check CSR structural invariants."""
    assert len(indptr) == n_rows + 1, f"indptr length {len(indptr)} != {n_rows + 1}"
    assert indptr[0] == 0, "indptr must start at 0"
    assert indptr[-1] == len(indices), "indptr[-1] must equal len(indices)"
    assert len(indices) == len(data), "indices and data length mismatch"
    for i in range(n_rows):
        assert indptr[i] <= indptr[i + 1], "indptr must be non-decreasing"
    assert np.all(indices >= 0), "negative index"
    assert np.all(indices < n_cols), f"index >= {n_cols}"


__all__ = [
    "Any",
    "cast",
    "np",
    "pytest",
    "random_connectivity",
    "small_world",
    "scale_free",
    "ring_topology",
    "grid_topology",
    "all_to_all",
    "_csr_to_dense",
    "_validate_csr",
]
