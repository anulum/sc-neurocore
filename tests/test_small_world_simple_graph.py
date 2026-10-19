# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Watts-Strogatz small world as a simple graph, and the grid metric

"""``small_world`` is a simple graph and validates its domain; ``grid_topology`` is Chebyshev.

Reproducers of RIGOR review 270: the rewired graph had a parallel pair (two
CSR entries for one connection, doubling the current a spike delivers), and
an impossible degree made the generator loop. The grid generator connects a
Chebyshev neighbourhood, which the documentation now states.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.network import topology


def _pairs(n: int, csr: tuple[np.ndarray, np.ndarray, np.ndarray]) -> list[tuple[int, int]]:
    """Return the (row, column) pairs of a CSR matrix, one per stored entry."""
    indptr, indices, _data = csr
    rows = np.repeat(np.arange(n), np.diff(indptr))
    return list(zip(rows.tolist(), indices.tolist()))


@pytest.mark.parametrize(
    ("n", "k", "p", "seed"), [(10, 4, 1.0, 42), (30, 4, 0.1, 42), (50, 6, 0.5, 3), (10, 4, 0.0, 42)]
)
def test_no_parallel_or_self_connections_and_n_k_entries(
    n: int, k: int, p: float, seed: int
) -> None:
    """Every stored entry is a distinct off-diagonal pair; the graph is symmetric with n·k entries."""
    pairs = _pairs(n, topology.small_world(n, k, p, 1.0, seed=seed))
    assert len(pairs) == len(set(pairs)) == n * k
    assert all(row != col for row, col in pairs)
    assert set(pairs) == {(col, row) for row, col in pairs}


def test_no_rewiring_is_the_ring_lattice() -> None:
    """Without rewiring the graph is exactly the k/2-nearest-neighbour ring, both ways."""
    n, k = 10, 4
    ring = {(i, (i + j) % n) for i in range(n) for j in (-2, -1, 1, 2)}
    assert set(_pairs(n, topology.small_world(n, k, 0.0, 1.0))) == ring


def test_one_spike_delivers_one_weight() -> None:
    """The review's reproducer: n=10, k=4, p=1, seed 42; every connection carries the weight once."""
    indptr, indices, data = topology.small_world(10, 4, 1.0, 1.0, seed=42)
    dense = np.zeros((10, 10))
    np.add.at(dense, (np.repeat(np.arange(10), np.diff(indptr)), indices), data)
    assert set(np.unique(dense)) <= {0.0, 1.0}


def test_seed_replays() -> None:
    """The same seed gives the same graph."""
    first = topology.small_world(30, 4, 0.3, 1.0, seed=9)
    second = topology.small_world(30, 4, 0.3, 1.0, seed=9)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize(
    ("n", "k", "p"),
    [(1, 2, 1.0), (10, 3, 0.1), (10, 10, 0.1), (10, -2, 0.1), (10, 4, 1.5), (0, 0, 0.0)],
)
def test_impossible_configurations_are_refused(n: int, k: int, p: float) -> None:
    """A degree that is odd, negative or not below n, a probability outside [0, 1] or n < 1."""
    with pytest.raises(ValueError):
        topology.small_world(n, k, p, 1.0)


def test_single_node_without_edges_is_valid() -> None:
    """A single node with degree 0 gives the empty graph."""
    indptr, indices, _data = topology.small_world(1, 0, 1.0, 1.0)
    assert indptr.tolist() == [0, 0] and indices.size == 0


def test_grid_is_a_symmetric_chebyshev_neighbourhood() -> None:
    """The centre of a 5×5 grid has 8 neighbours at radius 1 and 24 at radius 2."""
    for radius, expected in ((1, 8), (2, 24)):
        indptr, indices, data = topology.grid_topology(5, 5, radius, 1.0)
        assert indptr[13] - indptr[12] == expected
        pairs = _pairs(25, (indptr, indices, data))
        assert set(pairs) == {(col, row) for row, col in pairs}
