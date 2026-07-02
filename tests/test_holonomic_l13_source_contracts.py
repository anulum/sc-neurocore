# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific contracts for the L13 holonomic source adapter

"""Production contracts for the L13 holonomic source-field adapter."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.adapters.holonomic._jax_compat import jnp
from sc_neurocore.adapters.holonomic.l13_source import (
    L13_HolonomicParameters,
    L13_SourceAdapter,
)


def _adapter() -> L13_SourceAdapter:
    """Create a compact deterministic L13 adapter for contract tests."""
    return L13_SourceAdapter(
        L13_HolonomicParameters(
            n_vacuum_nodes=4,
            bitstream_length=6,
            j_primordial_coupling=0.0,
            h_potential_bias=0.0,
            lambda_scission=0.0,
        ),
        seed=13,
    )


def test_l13_scalar_feedback_broadcasts_to_all_vacuum_nodes() -> None:
    """Scalar L16 feedback is accepted and broadcast across the vacuum lattice."""
    adapter = _adapter()

    output = adapter.step_jax(0.05, inputs=jnp.asarray(1.0))

    assert output.shape == (4, 6)
    np.testing.assert_allclose(
        np.asarray(adapter.vacuum_state),
        np.full(4, float(np.asarray(adapter.vacuum_state)[0])),
        rtol=1e-7,
        atol=1e-7,
    )


def test_l13_mismatched_feedback_rows_broadcast_mean_drive() -> None:
    """Mismatched L16 feedback rows collapse to one deterministic mean drive."""
    adapter = _adapter()
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
        np.asarray(adapter.vacuum_state),
        np.full(4, float(np.asarray(adapter.vacuum_state)[0])),
        rtol=1e-7,
        atol=1e-7,
    )


def test_l13_vector_feedback_preserves_per_node_drive() -> None:
    """Rank-1 L16 feedback maps one value to each configured vacuum node."""
    adapter = _adapter()

    adapter.step_jax(0.05, inputs=jnp.array([0.0, 0.25, 0.75, 1.0], dtype=jnp.float32))

    vacuum = np.asarray(adapter.vacuum_state)
    assert vacuum[0] < vacuum[1] < vacuum[2] < vacuum[3]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_vacuum_nodes": cast(int, True)}, "n_vacuum_nodes"),
        ({"bitstream_length": cast(int, True)}, "bitstream_length"),
        ({"j_primordial_coupling": np.nan}, "j_primordial_coupling"),
        ({"h_potential_bias": np.inf}, "h_potential_bias"),
        ({"lambda_scission": np.nan}, "lambda_scission"),
        ({"lambda_scission": -0.01}, "lambda_scission"),
    ],
)
def test_l13_rejects_invalid_parameter_edges(kwargs: dict[str, Any], message: str) -> None:
    """Invalid L13 edge-case parameters fail before state allocation."""
    with pytest.raises(ValueError, match=message):
        L13_SourceAdapter(L13_HolonomicParameters(**kwargs))


@pytest.mark.parametrize("dt", [0.0, -0.01, np.inf, np.nan])
def test_l13_rejects_invalid_timestep_without_mutating_state(dt: float) -> None:
    """Invalid timesteps are rejected without changing vacuum or FIM state."""
    adapter = _adapter()
    before_vacuum = np.asarray(adapter.vacuum_state).copy()
    before_fim = np.asarray(adapter.fim_density).copy()

    with pytest.raises(ValueError, match="dt"):
        adapter.step_jax(dt, inputs=jnp.ones((4, 6), dtype=jnp.float32))

    np.testing.assert_array_equal(np.asarray(adapter.vacuum_state), before_vacuum)
    np.testing.assert_array_equal(np.asarray(adapter.fim_density), before_fim)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (jnp.array([], dtype=jnp.float32), "at least one value"),
        (jnp.ones((0, 6), dtype=jnp.float32), "at least one row"),
        (jnp.ones((2, 0), dtype=jnp.float32), "at least one column"),
        (jnp.ones((2, 3, 1), dtype=jnp.float32), "rank 0, 1, or 2"),
        (jnp.array([[1.0, np.nan, 0.0, 1.0]], dtype=jnp.float32), "finite values"),
    ],
)
def test_l13_rejects_invalid_feedback_without_mutating_state(inputs: Any, message: str) -> None:
    """Malformed L16 feedback fails before vacuum or FIM mutation."""
    adapter = _adapter()
    before_vacuum = np.asarray(adapter.vacuum_state).copy()
    before_fim = np.asarray(adapter.fim_density).copy()

    with pytest.raises(ValueError, match=message):
        adapter.step_jax(0.05, inputs=inputs)

    np.testing.assert_array_equal(np.asarray(adapter.vacuum_state), before_vacuum)
    np.testing.assert_array_equal(np.asarray(adapter.fim_density), before_fim)


def test_l13_decode_rejects_empty_bitstream_matrix() -> None:
    """Decode rejects empty bitstream matrices instead of returning NaN telemetry."""
    adapter = _adapter()

    with pytest.raises(ValueError, match="non-empty"):
        adapter.decode(jnp.zeros((0, 6), dtype=jnp.uint8))


@pytest.mark.parametrize(
    ("bitstreams", "message"),
    [
        (jnp.zeros((6,), dtype=jnp.uint8), "rank-2"),
        (jnp.array([[0.0, np.inf]], dtype=jnp.float32), "finite values"),
    ],
)
def test_l13_decode_rejects_malformed_bitstreams(bitstreams: Any, message: str) -> None:
    """Decode rejects malformed bitstreams before telemetry conversion."""
    adapter = _adapter()

    with pytest.raises(ValueError, match=message):
        adapter.decode(bitstreams)
