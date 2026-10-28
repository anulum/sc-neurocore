# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — topology builders refuse values outside their domain

"""Small-world and ring builders refuse non-integer counts and non-finite weights."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.network.topology import ring_topology, small_world


@pytest.mark.parametrize("n", [4.0, "4", True])
def test_small_world_refuses_a_non_integer_node_count(n: object) -> None:
    """``n`` must be an integer, not a float, a string or a bool."""
    with pytest.raises(ValueError, match="n must be an integer"):
        small_world(n, 2, 0.1, 1.0)  # type: ignore[arg-type]


def test_small_world_refuses_a_non_integer_degree() -> None:
    """``k`` is checked the same way as ``n``."""
    with pytest.raises(ValueError, match="k must be an integer"):
        small_world(6, 2.0, 0.1, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("weight", [math.inf, -math.inf, math.nan, "1.0", False])
def test_small_world_refuses_a_non_finite_weight(weight: object) -> None:
    """The weight must be a finite real number."""
    with pytest.raises(ValueError, match="weight must be finite"):
        small_world(6, 2, 0.1, weight)  # type: ignore[arg-type]


def test_a_complete_ring_has_nothing_to_rewire_to() -> None:
    """With every node joined to every other, each rewire is skipped and the graph stays complete."""
    indptr, indices, data = small_world(3, 2, 1.0, 0.5, seed=1)
    dense = np.zeros((3, 3))
    for row in range(3):
        dense[row, indices[indptr[row] : indptr[row + 1]]] = data[indptr[row] : indptr[row + 1]]
    expected = np.full((3, 3), 0.5)
    np.fill_diagonal(expected, 0.0)
    assert np.array_equal(dense, expected)


@pytest.mark.parametrize("n", [0, -3])
def test_ring_refuses_fewer_than_one_node(n: int) -> None:
    """A ring needs at least one node."""
    with pytest.raises(ValueError, match="n must be at least 1 for ring topology"):
        ring_topology(n, 0, 1.0)


@pytest.mark.parametrize("weight", [math.inf, math.nan, "2", True])
def test_ring_refuses_a_non_finite_weight(weight: object) -> None:
    """The ring weight must be a finite real number."""
    with pytest.raises(ValueError, match="weight must be finite"):
        ring_topology(5, 1, weight)  # type: ignore[arg-type]
