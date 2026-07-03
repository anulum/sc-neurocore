# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific contracts for the L6 holonomic planetary adapter

"""Production contracts for the L6 holonomic planetary adapter."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.adapters.holonomic._jax_compat import jnp
from sc_neurocore.adapters.holonomic.l6_plan import (
    L6_HolonomicParameters,
    L6_PlanetaryAdapter,
)


def test_l6_mismatched_region_count_broadcasts_mean_drive() -> None:
    """Mismatched upstream region counts collapse to one deterministic mean drive."""
    adapter = L6_PlanetaryAdapter(L6_HolonomicParameters(n_regions=4, bitstream_length=6))
    inputs = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    output = adapter.step_jax(0.05, inputs=inputs)

    assert output.shape == (4, 6)
    np.testing.assert_allclose(
        np.asarray(adapter.phi_planetary),
        np.full(4, float(np.asarray(adapter.phi_planetary)[0])),
        rtol=1e-7,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_regions": cast(int, True)}, "n_regions"),
        ({"bitstream_length": cast(int, True)}, "bitstream_length"),
        ({"f_schumann": np.nan}, "f_schumann"),
        ({"alpha_gaia": np.inf}, "alpha_gaia"),
        ({"p_percolation": np.nan}, "p_percolation"),
    ],
)
def test_l6_rejects_invalid_parameter_edges(kwargs: dict[str, Any], message: str) -> None:
    """Invalid L6 edge-case parameters fail before state allocation."""
    with pytest.raises(ValueError, match=message):
        L6_PlanetaryAdapter(L6_HolonomicParameters(**kwargs))


@pytest.mark.parametrize("dt", [0.0, -0.01, np.inf, np.nan])
def test_l6_rejects_invalid_timestep_without_mutating_state(dt: float) -> None:
    """Invalid timesteps are rejected without changing time or field state."""
    adapter = L6_PlanetaryAdapter(L6_HolonomicParameters(n_regions=3, bitstream_length=4))
    before_phi = np.asarray(adapter.phi_planetary).copy()
    before_coherence = np.asarray(adapter.regional_coherence).copy()

    with pytest.raises(ValueError, match="dt"):
        adapter.step_jax(dt, inputs=jnp.ones((3, 4), dtype=jnp.float32))

    assert adapter.t == 0.0
    np.testing.assert_array_equal(np.asarray(adapter.phi_planetary), before_phi)
    np.testing.assert_array_equal(np.asarray(adapter.regional_coherence), before_coherence)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (jnp.ones((0, 4), dtype=jnp.float32), "at least one row"),
        (jnp.ones((2, 3), dtype=jnp.float32), "bitstream_length"),
        (jnp.ones((4,), dtype=jnp.float32), "rank-2"),
        (jnp.array([[1.0, np.nan, 0.0, 1.0]], dtype=jnp.float32), "finite values"),
    ],
)
def test_l6_rejects_invalid_input_shape_without_mutating_state(inputs: Any, message: str) -> None:
    """Malformed upstream bitstream batches fail before time or field mutation."""
    adapter = L6_PlanetaryAdapter(L6_HolonomicParameters(n_regions=3, bitstream_length=4))
    before_phi = np.asarray(adapter.phi_planetary).copy()
    before_coherence = np.asarray(adapter.regional_coherence).copy()

    with pytest.raises(ValueError, match=message):
        adapter.step_jax(0.05, inputs=inputs)

    assert adapter.t == 0.0
    np.testing.assert_array_equal(np.asarray(adapter.phi_planetary), before_phi)
    np.testing.assert_array_equal(np.asarray(adapter.regional_coherence), before_coherence)


def test_l6_step_without_inputs_uses_zero_sync_drive() -> None:
    """Absent upstream bitstreams advance time through a zero synchrony drive."""
    adapter = L6_PlanetaryAdapter(L6_HolonomicParameters(n_regions=3, bitstream_length=4))

    output = adapter.step_jax(0.05)

    assert output.shape == (3, 4)
    assert adapter.t == pytest.approx(0.05)
    np.testing.assert_array_equal(np.asarray(adapter.phi_planetary), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(adapter.regional_coherence), np.zeros(3))
    np.testing.assert_array_equal(np.asarray(output), np.zeros((3, 4), dtype=np.uint8))


def test_l6_decode_reports_global_coherence_mean() -> None:
    """Decode returns the mean bitstream occupancy as global coherence."""
    adapter = L6_PlanetaryAdapter(L6_HolonomicParameters(n_regions=2, bitstream_length=4))
    bitstreams = jnp.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=jnp.uint8,
    )

    telemetry = adapter.decode(bitstreams)

    assert telemetry["global_coherence_index"] == pytest.approx(0.75)
