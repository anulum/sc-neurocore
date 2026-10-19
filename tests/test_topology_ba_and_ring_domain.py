# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Barabasi-Albert attaches by actual degree; a ring is a simple graph

"""Scale-free graphs attach by actual degree, and rings stay simple.

Reproducers of RIGOR review 282: ``scale_free`` gave every seed node a degree
of 1 without emitting an edge and then counted its later edges on top, so the
attachment kernel was not the actual degree (two seed nodes of degree 1 each
were drawn 2/3 : 1/3); and ``ring_topology`` accepted ``k = n / 2``, doubling
the antipodal connection, and ``k >= n``, adding self connections.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sc_neurocore.network.topology import ring_topology, scale_free


def _dense(
    csr: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]], n: int
) -> np.ndarray[Any, Any]:
    """Return the dense adjacency count matrix of a CSR triple."""
    indptr, indices, _data = csr
    dense = np.zeros((n, n), dtype=np.int64)
    for row in range(n):
        for k in range(indptr[row], indptr[row + 1]):
            dense[row, indices[k]] += 1
    return dense


@pytest.mark.parametrize(("n", "m"), [(2, 1), (3, 1), (4, 2), (12, 3), (30, 2)])
def test_a_scale_free_graph_is_simple_and_counts_its_edges(n: int, m: int) -> None:
    """Symmetric, no self or parallel edges; the star seed and m edges per later node."""
    dense = _dense(scale_free(n, m, 1.0, seed=4), n)
    assert np.array_equal(dense, dense.T)
    assert dense.max() <= 1 and not np.any(np.diag(dense))
    assert int(dense.sum()) // 2 == m + m * (n - m - 1)


def test_attachment_follows_the_actual_degree() -> None:
    """n=3, m=1: the seed is the edge 0-1, both of degree 1, so node 2 joins each half the time.

    The former kernel gave node 0 two thirds. Over 400 seeds the share of node 0
    lies within about four standard deviations (0.025) of one half.
    """
    joined_zero = 0
    for seed in range(400):
        dense = _dense(scale_free(3, 1, 1.0, seed=seed), 3)
        joined_zero += int(dense[2, 0])
    assert 0.40 <= joined_zero / 400 <= 0.60


@pytest.mark.parametrize(("n", "k"), [(6, 3), (6, 6), (1, 1), (4, 2), (5, 3)])
def test_a_ring_that_would_double_or_self_connect_is_refused(n: int, k: int) -> None:
    """A k above (n - 1) // 2 meets the antipode twice or reaches the node itself."""
    with pytest.raises(ValueError, match="simple ring"):
        ring_topology(n, k, 1.0)


@pytest.mark.parametrize(("n", "k"), [(6, 2), (7, 3), (1, 0), (50, 3)])
def test_a_ring_in_its_domain_is_simple(n: int, k: int) -> None:
    """2k distinct neighbours per node, symmetric, no self connection (k=2 at n=6 the control)."""
    dense = _dense(ring_topology(n, k, 1.0), n)
    assert np.array_equal(dense, dense.T)
    assert dense.max(initial=0) <= 1 and not np.any(np.diag(dense))
    assert dense.sum(axis=1).tolist() == [2 * k] * n


@pytest.mark.parametrize("bad", [2.0, True, -1])
def test_ring_arguments_are_validated(bad: Any) -> None:
    """A non-integer, boolean or negative neighbour count is refused."""
    with pytest.raises(ValueError):
        ring_topology(6, bad, 1.0)
