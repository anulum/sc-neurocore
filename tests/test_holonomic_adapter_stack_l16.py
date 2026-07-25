# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused holonomic adapter contracts

from __future__ import annotations

from tests.holonomic_adapter_stack_contract_support import *  # noqa: F403


def test_l16_adapter_init_and_step():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    params = L16_HolonomicParameters(n_meta_nodes=4, bitstream_length=64)
    adapter = L16_MetaAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (4, 64)
    metrics = adapter.get_metrics()
    assert "director_will" in metrics
    assert "system_entropy" in metrics
    assert "veto_active" in metrics
    decoded = adapter.decode(out)
    assert "meta_coherence_r16" in decoded


def test_l16_with_gci_input():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    adapter = L16_MetaAdapter(L16_HolonomicParameters(n_meta_nodes=4, bitstream_length=32))
    inputs = jnp.ones((1, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (4, 32)


def test_l16_veto_activation():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    params = L16_HolonomicParameters(n_meta_nodes=4, veto_threshold=0.1)
    adapter = L16_MetaAdapter(params)
    adapter.entropy_proxy = 0.9
    adapter.step_jax(0.1)
    assert float(jnp.mean(adapter.veto_active)) > 0.0
