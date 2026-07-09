# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


"""Module-specific contracts for the L7 holonomic symbolic adapter."""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pytest

from sc_neurocore.adapters.holonomic._jax_compat import jnp
from sc_neurocore.adapters.holonomic.l7_sym import (
    L7_HolonomicParameters,
    L7_SymbolicAdapter,
)


def test_l7_single_node_topology_is_identity_router() -> None:
    """Single-node L7 topologies use a stable identity routing matrix."""
    adapter = L7_SymbolicAdapter(L7_HolonomicParameters(n_nodes=1, bitstream_length=8))

    matrix = np.asarray(adapter.metatron_matrix)

    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(1.0)
    output = adapter.step_jax(0.01)
    assert output.shape == (1, 8)


def test_l7_nonstandard_node_count_uses_ring_topology() -> None:
    """Non-13-node L7 topologies use the documented centre-plus-ring fallback."""
    coords = L7_SymbolicAdapter._metatron_coordinates(5)

    assert coords.shape == (5, 2)
    np.testing.assert_allclose(coords[0], np.zeros(2), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.linalg.norm(coords[1:], axis=1), np.ones(4))


def test_l7_rejects_underflowed_metatron_topology() -> None:
    """Topology initialisation fail-closes when all off-diagonal weights vanish."""
    params = L7_HolonomicParameters(n_nodes=2, bitstream_length=8, phi_golden_ratio=1e-320)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="off-diagonal edge"):
            L7_SymbolicAdapter(params)

    assert caught == []


def test_l7_rejects_boolean_integer_parameters() -> None:
    """Boolean values are rejected even though bool is an int subclass."""
    with pytest.raises(ValueError, match="n_nodes"):
        L7_SymbolicAdapter(L7_HolonomicParameters(n_nodes=cast(int, True)))
    with pytest.raises(ValueError, match="bitstream_length"):
        L7_SymbolicAdapter(L7_HolonomicParameters(bitstream_length=cast(int, True)))


@pytest.mark.parametrize(
    "phi_golden_ratio",
    [0.0, np.inf, np.nan],
)
def test_l7_rejects_invalid_phi_golden_ratio(phi_golden_ratio: float) -> None:
    """The geometric length scale must be finite and positive."""
    with pytest.raises(ValueError, match="phi_golden_ratio"):
        L7_SymbolicAdapter(L7_HolonomicParameters(phi_golden_ratio=phi_golden_ratio))


def test_l7_mismatched_input_width_broadcasts_mean_drive() -> None:
    """Mismatched upstream width collapses to a mean drive over all L7 nodes."""
    params = L7_HolonomicParameters(n_nodes=4, bitstream_length=8)
    adapter = L7_SymbolicAdapter(params)
    inputs = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=jnp.float32,
    )

    output = adapter.step_jax(0.1, inputs=inputs)

    assert output.shape == (4, 8)
    np.testing.assert_allclose(
        np.asarray(adapter.node_phases),
        np.full(4, 0.05),
        rtol=1e-6,
        atol=1e-6,
    )
