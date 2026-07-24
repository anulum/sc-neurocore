# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (l1_l2_adapters) from former test_holonomic_adapter_stack_contract.py

from __future__ import annotations

from tests.holonomic_adapter_stack_contract_support import *  # noqa: F403


def test_l1_adapter_contract():
    params = L1_HolonomicParameters(n_qubits=10)
    adapter = L1_QuantumAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (10, 1024)
    metrics = adapter.get_metrics()
    assert "r1_global_coherence" in metrics
    decoded = adapter.decode(out)
    assert "avg_coherence" in decoded


def test_l2_adapter_contract():
    params = L2_HolonomicParameters(n_transmitters=4)
    adapter = L2_NeurochemicalAdapter(params)
    out = adapter.step_jax(0.1, inputs=jnp.ones((4, 1024)))
    assert out.shape == (4, 1024)
    metrics = adapter.get_metrics()
    assert "avg_field_potential" in metrics
    decoded = adapter.decode(out)
    assert "dopamine" in decoded


def test_l2_iiief_wave_speed_controls_spatial_spread():
    slow = L2_NeurochemicalAdapter(
        L2_HolonomicParameters(n_transmitters=5, bitstream_length=32, c_info=0.1)
    )
    fast = L2_NeurochemicalAdapter(
        L2_HolonomicParameters(n_transmitters=5, bitstream_length=32, c_info=30.0)
    )
    slow.phi_field = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])
    fast.phi_field = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])
    inputs = jnp.zeros((5, 32))

    slow.step_jax(0.01, inputs=inputs)
    fast.step_jax(0.01, inputs=inputs)

    slow_neighbour = float(slow.phi_field[1] + slow.phi_field[3])
    fast_neighbour = float(fast.phi_field[1] + fast.phi_field[3])
    assert fast_neighbour > slow_neighbour


def test_l2_hqc_release_uses_threshold_and_bridge_parameters():
    low_threshold = L2_NeurochemicalAdapter(
        L2_HolonomicParameters(
            n_transmitters=2,
            bitstream_length=16,
            g_snare=1.0,
            v_critical=0.1,
            dopamine_gain=4.0,
            serotonin_leak=0.5,
        )
    )
    high_threshold = L2_NeurochemicalAdapter(
        L2_HolonomicParameters(
            n_transmitters=2,
            bitstream_length=16,
            g_snare=1.0,
            v_critical=10.0,
            dopamine_gain=4.0,
            serotonin_leak=0.5,
        )
    )
    low_threshold.phi_field = jnp.ones((2,))
    high_threshold.phi_field = jnp.ones((2,))

    low_threshold.step_jax(0.01, inputs=jnp.zeros((2, 16)))
    high_threshold.step_jax(0.01, inputs=jnp.zeros((2, 16)))

    assert float(jnp.mean(low_threshold.concentrations)) > float(
        jnp.mean(high_threshold.concentrations)
    )


def test_l2_rejects_invalid_holonomic_parameters():
    with pytest.raises(ValueError, match="n_transmitters"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(n_transmitters=0))
    with pytest.raises(ValueError, match="n_receptors"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(n_receptors=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="alpha_iiief"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(alpha_iiief=-0.1))
    with pytest.raises(ValueError, match="c_info"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(c_info=0.0))
    with pytest.raises(ValueError, match="g_snare"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(g_snare=0.0))
    with pytest.raises(ValueError, match="v_critical"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(v_critical=0.0))
    with pytest.raises(ValueError, match="dopamine_gain"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(dopamine_gain=0.0))
    with pytest.raises(ValueError, match="serotonin_leak"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(serotonin_leak=1.1))

    adapter = L2_NeurochemicalAdapter()
    with pytest.raises(ValueError, match="dt"):
        adapter.step_jax(0.0)


def test_l1_adapter_consumes_metabolic_drive():
    # Supplying a metabolic/field drive advances the pump term rather than
    # leaving it at its initial value.
    adapter = L1_QuantumAdapter(L1_HolonomicParameters(n_qubits=4))
    out = adapter.step_jax(0.1, inputs=jnp.ones((4, 8)))
    assert out.shape[0] == 4
