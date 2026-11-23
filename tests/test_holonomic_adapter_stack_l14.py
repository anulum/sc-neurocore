# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused holonomic adapter contracts

"""Contract tests of the L14 holonomic adapter."""

from __future__ import annotations

from sc_neurocore.accel.jax_backend import (
    jnp,
)
from sc_neurocore.adapters.holonomic.l14_trans import (
    L14_HolonomicParameters,
    L14_TransdimensionalAdapter,
)


def test_l14_adapter_init_and_step() -> None:
    """The L14 adapter steps and reports its alignment metrics."""
    params = L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=64)
    adapter = L14_TransdimensionalAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (5, 64)
    metrics = adapter.get_metrics()
    assert "avg_brane_alignment" in metrics
    assert "resonance_sharpness" in metrics
    decoded = adapter.decode(out)
    assert "brane_resonance_r14" in decoded


def test_l14_with_inputs() -> None:
    """The L14 adapter takes an input stream."""
    adapter = L14_TransdimensionalAdapter(
        L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=32)
    )
    inputs = jnp.ones((5, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (5, 32)


def test_l14_input_broadcast() -> None:
    """The L14 adapter broadcasts its input."""
    adapter = L14_TransdimensionalAdapter(
        L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=32)
    )
    inputs = jnp.ones((3, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (5, 32)
