# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused holonomic adapter contracts

from __future__ import annotations

from tests.holonomic_adapter_stack_contract_support import *  # noqa: F403


def test_l15_adapter_init_and_step():  # type: ignore[no-untyped-def] # Preserved legacy test AST
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


def test_l15_with_inputs():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    adapter = L15_ConsiliumAdapter(
        L15_HolonomicParameters(n_metric_dimensions=8, bitstream_length=32)
    )
    inputs = jnp.ones((8, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (8, 32)


def test_l15_partial_stack_padding():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    adapter = L15_ConsiliumAdapter(
        L15_HolonomicParameters(n_metric_dimensions=16, bitstream_length=32)
    )
    inputs = jnp.ones((4, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (16, 32)
