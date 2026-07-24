# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (l13_l16_adapters) from former test_holonomic_adapter_stack_contract.py

from __future__ import annotations

from tests.holonomic_adapter_stack_contract_support import *  # noqa: F403

def test_l13_adapter_init_and_step():
    params = L13_HolonomicParameters(n_vacuum_nodes=8, bitstream_length=64)
    adapter = L13_SourceAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (8, 64)
    metrics = adapter.get_metrics()
    assert "vacuum_potential" in metrics
    assert "fisher_information_metric" in metrics
    decoded = adapter.decode(out)
    assert "source_coherence_r13" in decoded


def test_l13_encode():
    adapter = L13_SourceAdapter(L13_HolonomicParameters(n_vacuum_nodes=4, bitstream_length=32))
    bits = adapter.encode(None)
    assert bits.shape == (4, 32)
    assert bits.dtype == jnp.uint8


def test_l13_vacuum_kernel_clip():
    adapter = L13_SourceAdapter(L13_HolonomicParameters(n_vacuum_nodes=4))
    for _ in range(50):
        adapter.step_jax(0.5)
    state = adapter.vacuum_state
    assert float(jnp.min(state)) >= 0.0
    assert float(jnp.max(state)) <= 1.0


def test_l13_vacuum_kernel_uses_local_lattice_coupling():
    adapter = L13_SourceAdapter(
        L13_HolonomicParameters(
            n_vacuum_nodes=7,
            bitstream_length=32,
            j_primordial_coupling=1.0,
            h_potential_bias=0.0,
            lambda_scission=0.0,
        )
    )
    adapter.vacuum_state = jnp.array([0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5])

    adapter.step_jax(0.05)

    neighbour_lift = float((adapter.vacuum_state[2] - 0.5) + (adapter.vacuum_state[4] - 0.5))
    far_lift = float((adapter.vacuum_state[0] - 0.5) + (adapter.vacuum_state[6] - 0.5))
    assert neighbour_lift > far_lift


def test_l13_scission_rate_controls_symmetry_breaking():
    no_scission = L13_SourceAdapter(
        L13_HolonomicParameters(n_vacuum_nodes=16, bitstream_length=16, lambda_scission=0.0),
        seed=77,
    )
    active_scission = L13_SourceAdapter(
        L13_HolonomicParameters(n_vacuum_nodes=16, bitstream_length=16, lambda_scission=0.8),
        seed=77,
    )

    no_scission.step_jax(0.01)
    active_scission.step_jax(0.01)

    assert float(jnp.mean(jnp.abs(active_scission.vacuum_state - 0.5))) > float(
        jnp.mean(jnp.abs(no_scission.vacuum_state - 0.5))
    )


def test_l13_l16_feedback_modulates_vacuum_potential():
    baseline = L13_SourceAdapter(
        L13_HolonomicParameters(
            n_vacuum_nodes=4,
            bitstream_length=16,
            j_primordial_coupling=0.0,
            h_potential_bias=0.0,
            lambda_scission=0.0,
        )
    )
    driven = L13_SourceAdapter(
        L13_HolonomicParameters(
            n_vacuum_nodes=4,
            bitstream_length=16,
            j_primordial_coupling=0.0,
            h_potential_bias=0.0,
            lambda_scission=0.0,
        )
    )

    baseline.step_jax(0.05, inputs=jnp.zeros((4, 16)))
    driven.step_jax(0.05, inputs=jnp.ones((4, 16)))

    assert float(jnp.mean(driven.vacuum_state)) > float(jnp.mean(baseline.vacuum_state))


def test_l13_rejects_invalid_holonomic_parameters_and_dt():
    with pytest.raises(ValueError, match="n_vacuum_nodes"):
        L13_SourceAdapter(L13_HolonomicParameters(n_vacuum_nodes=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L13_SourceAdapter(L13_HolonomicParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="j_primordial_coupling"):
        L13_SourceAdapter(L13_HolonomicParameters(j_primordial_coupling=float("nan")))
    with pytest.raises(ValueError, match="h_potential_bias"):
        L13_SourceAdapter(L13_HolonomicParameters(h_potential_bias=float("inf")))
    with pytest.raises(ValueError, match="lambda_scission"):
        L13_SourceAdapter(L13_HolonomicParameters(lambda_scission=-0.1))

    adapter = L13_SourceAdapter()
    with pytest.raises(ValueError, match="dt"):
        adapter.step_jax(0.0)
    with pytest.raises(ValueError, match="inputs"):
        adapter.step_jax(0.01, inputs=jnp.array([float("nan")]))


def test_l14_adapter_init_and_step():
    params = L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=64)
    adapter = L14_TransdimensionalAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (5, 64)
    metrics = adapter.get_metrics()
    assert "avg_brane_alignment" in metrics
    assert "resonance_sharpness" in metrics
    decoded = adapter.decode(out)
    assert "brane_resonance_r14" in decoded


def test_l14_with_inputs():
    adapter = L14_TransdimensionalAdapter(
        L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=32)
    )
    inputs = jnp.ones((5, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (5, 32)


def test_l14_input_broadcast():
    adapter = L14_TransdimensionalAdapter(
        L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=32)
    )
    inputs = jnp.ones((3, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (5, 32)


def test_l15_adapter_init_and_step():
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


def test_l15_with_inputs():
    adapter = L15_ConsiliumAdapter(
        L15_HolonomicParameters(n_metric_dimensions=8, bitstream_length=32)
    )
    inputs = jnp.ones((8, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (8, 32)


def test_l15_partial_stack_padding():
    adapter = L15_ConsiliumAdapter(
        L15_HolonomicParameters(n_metric_dimensions=16, bitstream_length=32)
    )
    inputs = jnp.ones((4, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (16, 32)


def test_l16_adapter_init_and_step():
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


def test_l16_with_gci_input():
    adapter = L16_MetaAdapter(L16_HolonomicParameters(n_meta_nodes=4, bitstream_length=32))
    inputs = jnp.ones((1, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (4, 32)


def test_l16_veto_activation():
    params = L16_HolonomicParameters(n_meta_nodes=4, veto_threshold=0.1)
    adapter = L16_MetaAdapter(params)
    adapter.entropy_proxy = 0.9
    adapter.step_jax(0.1)
    assert float(jnp.mean(adapter.veto_active)) > 0.0


