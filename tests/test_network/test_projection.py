# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Projection connectivity, delay, and plasticity branch tests

"""Targeted branch coverage for ``network.projection``.

The CSR validator, the delay-mode accessors, the topology builder, the
weight-threshold matvec skip, the defensive uninitialised-state guards, and the
self-projection symmetry enforcement are exercised directly here; the happy-path
matvec and delayed propagation are covered transitively by the model suite.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection, validate_csr_topology


def _pops(n: int = 2) -> tuple[Population, Population]:
    return Population("LapicqueNeuron", n), Population("LapicqueNeuron", n)


# ── validate_csr_topology ────────────────────────────────────────────


def test_validate_csr_topology_accepts_and_normalises_valid_arrays() -> None:
    """Check that valid CSR arrays are returned as int64, int64 and float64 arrays."""
    indptr, indices, data = validate_csr_topology(
        np.array([0, 1, 2]), np.array([1, 0]), np.array([0.5, 0.25]), 2, 2
    )
    assert indptr.dtype == np.int64
    assert indices.dtype == np.int64
    assert data.dtype == np.float64


@pytest.mark.parametrize(
    "indptr,indices,data,match",
    [
        (np.zeros((2, 2), int), np.array([0]), np.array([1.0]), "indptr must be 1-D"),
        (np.array([0, 1]), np.zeros((1, 1), int), np.array([1.0]), "indices must be 1-D"),
        (np.array([0, 1]), np.array([0]), np.zeros((1, 1)), "data must be 1-D"),
        (np.array([0.0, 1.0]), np.array([0]), np.array([1.0]), "indptr must contain integers"),
        (np.array([0, 1]), np.array([0.0]), np.array([1.0]), "indices must contain integers"),
        (np.array([0, 1, 2, 3]), np.array([0]), np.array([1.0]), "indptr length is invalid"),
        (np.array([0, 1, 2]), np.array([0, 1]), np.array([1.0]), "indices/data lengths differ"),
        (np.array([1, 2, 3]), np.array([0, 1]), np.array([1.0, 1.0]), "indptr must start at 0"),
        (np.array([0, 2, 1]), np.array([0, 1]), np.array([1.0, 1.0]), "indptr must be monotonic"),
        (np.array([0, 1, 1]), np.array([0, 1]), np.array([1.0, 1.0]), "terminal length is invalid"),
        (np.array([0, 1, 2]), np.array([0, 9]), np.array([1.0, 1.0]), "indices are out of bounds"),
        (
            np.array([0, 1, 2]),
            np.array([0, 1]),
            np.array([1.0, np.inf]),
            "data must contain only finite values",
        ),
    ],
)
def test_validate_csr_topology_rejects_malformed_arrays(
    indptr: np.ndarray[Any, Any],
    indices: np.ndarray[Any, Any],
    data: np.ndarray[Any, Any],
    match: str,
) -> None:
    """Check that each malformed CSR input raises ValueError with its specific message."""
    with pytest.raises(ValueError, match=match):
        validate_csr_topology(indptr, indices, data, 2, 2)


# ── topology builder ─────────────────────────────────────────────────


def test_prebuilt_csr_tuple_topology_is_validated() -> None:
    """Check that a prebuilt CSR tuple topology is accepted and gives two synapses."""
    src, tgt = _pops(2)
    csr = (np.array([0, 1, 2]), np.array([1, 0]), np.array([0.5, 0.5]))
    proj = Projection(src, tgt, weight=0.5, topology=csr)
    assert proj.n_synapses == 2


def test_same_size_only_topologies_reject_without_prebuilt_csr() -> None:
    """Check that the ring topology raises its same-size requirement error without a prebuilt CSR."""
    src, tgt = _pops(2)
    with pytest.raises(ValueError, match="requires same-size source/target"):
        Projection(src, tgt, weight=0.5, topology="ring")


def test_unknown_topology_name_raises() -> None:
    """Check that an unrecognised topology name raises ValueError."""
    src, tgt = _pops(2)
    with pytest.raises(ValueError, match="Unknown topology"):
        Projection(src, tgt, weight=0.5, topology="banana")


# ── delay-mode accessors ─────────────────────────────────────────────


def test_max_delay_none_mode_is_zero() -> None:
    """Check that a zero delay selects the 'none' mode with max_delay 0."""
    src, tgt = _pops(2)
    proj = Projection(src, tgt, weight=0.5, topology="all_to_all", delay=0.0)
    assert proj.delay_mode == "none"
    assert proj.max_delay == 0


def test_max_delay_uniform_mode_returns_steps() -> None:
    """Check that a scalar delay of 2 selects uniform mode with max_delay 2."""
    src, tgt = _pops(2)
    proj = Projection(src, tgt, weight=0.5, topology="all_to_all", delay=2.0)
    assert proj.delay_mode == "uniform"
    assert proj.max_delay == 2


def test_max_delay_per_synapse_without_state_raises() -> None:
    """Check that max_delay raises RuntimeError when per-synapse delays are missing."""
    src, tgt = _pops(2)
    proj = Projection(src, tgt, weight=0.5, topology="all_to_all", delay=np.full(4, 3.0))
    proj._per_syn_delays = None  # simulate corrupted delay state
    with pytest.raises(RuntimeError, match="per-synapse delay state is not initialized"):
        _ = proj.max_delay


def test_delay_steps_per_synapse_without_state_raises() -> None:
    """Check that delay_steps raises RuntimeError when per-synapse delays are missing."""
    src, tgt = _pops(2)
    proj = Projection(src, tgt, weight=0.5, topology="all_to_all", delay=np.full(4, 3.0))
    proj._per_syn_delays = None  # simulate corrupted delay state
    with pytest.raises(RuntimeError, match="per-synapse delay state is not initialized"):
        _ = proj.delay_steps


# ── propagation ──────────────────────────────────────────────────────


def test_weight_threshold_skips_subthreshold_synapses() -> None:
    """Check that weights at or below the threshold contribute no current."""
    src, tgt = _pops(2)
    proj = Projection(src, tgt, weight=0.5, topology="all_to_all", weight_threshold=1.0)
    out = proj.propagate(np.array([1, 1], dtype=np.float64))
    # Every weight (0.5) is at/below the threshold, so no current accumulates.
    assert np.all(out == 0.0)


def test_propagate_uniform_without_buffer_raises() -> None:
    """Check that propagate raises RuntimeError when the uniform delay buffer is missing."""
    src, tgt = _pops(2)
    proj = Projection(src, tgt, weight=0.5, topology="all_to_all", delay=2.0)
    proj._delay_buf = None  # simulate corrupted delay buffer
    with pytest.raises(RuntimeError, match="uniform delay buffer is not initialized"):
        proj.propagate(np.array([1, 1], dtype=np.float64))


def test_propagate_per_synapse_without_state_raises() -> None:
    """Check that propagate raises RuntimeError when per-synapse delays are missing."""
    src, tgt = _pops(2)
    proj = Projection(src, tgt, weight=0.5, topology="all_to_all", delay=np.full(4, 3.0))
    proj._per_syn_delays = None  # simulate corrupted delay state
    with pytest.raises(RuntimeError, match="per-synapse delay state is not initialized"):
        proj.propagate(np.array([1, 1], dtype=np.float64))


# ── plasticity ───────────────────────────────────────────────────────


def test_update_plasticity_noop_without_stdp() -> None:
    """Check that update_plasticity leaves weights unchanged without a plasticity rule."""
    src, tgt = _pops(2)
    proj = Projection(src, tgt, weight=0.5, topology="all_to_all")  # plasticity None
    before = proj.data.copy()
    proj.update_plasticity(np.array([1, 1]), np.array([1, 1]))
    np.testing.assert_array_equal(proj.data, before)


def test_self_projection_stdp_enforces_weight_symmetry() -> None:
    """Check that STDP on a self-projection keeps reverse-edge weights equal."""
    src = Population("LapicqueNeuron", 2)
    proj = Projection(src, src, weight=0.5, topology="all_to_all", plasticity="stdp")
    proj.update_plasticity(np.array([1, 1]), np.array([1, 1]))
    # After enforcing symmetry, reverse-edge weights match (W_ij == W_ji).
    dense = np.zeros((2, 2))
    for i in range(2):
        for k in range(proj.indptr[i], proj.indptr[i + 1]):
            dense[i, proj.indices[k]] = proj.data[k]
    assert dense[0, 1] == pytest.approx(dense[1, 0])
