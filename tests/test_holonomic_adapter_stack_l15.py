# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused holonomic adapter contracts

"""Contract tests of the L15 holonomic adapter."""

from __future__ import annotations

from sc_neurocore.accel.jax_backend import (
    jnp,
)
from sc_neurocore.adapters.holonomic.l15_cons import (
    L15_ConsiliumAdapter,
    L15_HolonomicParameters,
)


def test_l15_adapter_init_and_step() -> None:
    """The L15 adapter steps and reports its GCI metrics."""
    params = L15_HolonomicParameters(n_metric_dimensions=8, bitstream_length=64)
    adapter = L15_ConsiliumAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (8, 64)
    metrics = adapter.get_metrics()
    assert "gci_index" in metrics
    assert "metric_entropy" in metrics
    assert "optimizer_error" in metrics
    decoded = adapter.decode(out)
    assert "global_coherence_r15" in decoded


def test_l15_with_inputs() -> None:
    """The L15 adapter takes an input stream."""
    adapter = L15_ConsiliumAdapter(
        L15_HolonomicParameters(n_metric_dimensions=8, bitstream_length=32)
    )
    inputs = jnp.ones((8, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (8, 32)


def test_l15_partial_stack_padding() -> None:
    """A partial stack is padded to full height."""
    adapter = L15_ConsiliumAdapter(
        L15_HolonomicParameters(n_metric_dimensions=16, bitstream_length=32)
    )
    inputs = jnp.ones((4, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (16, 32)
