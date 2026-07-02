# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific contracts for the L9 holonomic memory adapter

"""Production contracts for the L9 holonomic memory adapter."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.adapters.holonomic._jax_compat import jnp
from sc_neurocore.adapters.holonomic.l9_mem import (
    L9_HolonomicParameters,
    L9_MemoryAdapter,
)


def test_l9_tiles_mismatched_input_slots_deterministically() -> None:
    """Mismatched upstream slot counts tile over all configured memory slots."""
    params = L9_HolonomicParameters(n_memory_slots=4, bitstream_length=6)
    adapter = L9_MemoryAdapter(params, seed=7)
    inputs = jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    output = adapter.step_jax(0.05, inputs=inputs)

    assert output.shape == (6,)
    np.testing.assert_array_equal(
        np.asarray(adapter.imprints_psi),
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_memory_slots": 0}, "n_memory_slots"),
        ({"n_memory_slots": cast(int, True)}, "n_memory_slots"),
        ({"bitstream_length": 0}, "bitstream_length"),
        ({"bitstream_length": cast(int, True)}, "bitstream_length"),
        ({"retrieval_gain": np.nan}, "retrieval_gain"),
        ({"retrieval_gain": -0.1}, "retrieval_gain"),
        ({"weak_measurement_strength": np.inf}, "weak_measurement_strength"),
        ({"weak_measurement_strength": -0.1}, "weak_measurement_strength"),
        ({"temporal_window": 0}, "temporal_window"),
        ({"temporal_window": cast(int, True)}, "temporal_window"),
    ],
)
def test_l9_rejects_invalid_parameters(kwargs: dict[str, Any], message: str) -> None:
    """Invalid L9 configuration values fail before state allocation."""
    with pytest.raises(ValueError, match=message):
        L9_MemoryAdapter(L9_HolonomicParameters(**kwargs))


@pytest.mark.parametrize("dt", [0.0, -0.1, np.inf, np.nan])
def test_l9_rejects_invalid_timestep_without_mutating_state(dt: float) -> None:
    """Invalid timesteps are rejected before TSVF state mutation."""
    adapter = L9_MemoryAdapter(L9_HolonomicParameters(n_memory_slots=2, bitstream_length=4))
    before_psi = np.asarray(adapter.imprints_psi).copy()
    before_phi = np.asarray(adapter.retrieval_phi).copy()

    with pytest.raises(ValueError, match="dt"):
        adapter.step_jax(dt, inputs=jnp.ones((2, 4), dtype=jnp.float32))

    np.testing.assert_array_equal(np.asarray(adapter.imprints_psi), before_psi)
    np.testing.assert_array_equal(np.asarray(adapter.retrieval_phi), before_phi)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        (jnp.ones((0, 4), dtype=jnp.float32), "at least one row"),
        (jnp.ones((2, 3), dtype=jnp.float32), "bitstream_length"),
        (jnp.ones((4,), dtype=jnp.float32), "rank-2"),
        (jnp.array([[1.0, np.nan, 0.0, 1.0]], dtype=jnp.float32), "finite values"),
    ],
)
def test_l9_rejects_invalid_input_shape_without_mutating_state(inputs: Any, message: str) -> None:
    """Malformed upstream bitstream batches fail before TSVF state mutation."""
    adapter = L9_MemoryAdapter(L9_HolonomicParameters(n_memory_slots=2, bitstream_length=4))
    before_psi = np.asarray(adapter.imprints_psi).copy()
    before_phi = np.asarray(adapter.retrieval_phi).copy()

    with pytest.raises(ValueError, match=message):
        adapter.step_jax(0.05, inputs=inputs)

    np.testing.assert_array_equal(np.asarray(adapter.imprints_psi), before_psi)
    np.testing.assert_array_equal(np.asarray(adapter.retrieval_phi), before_phi)
