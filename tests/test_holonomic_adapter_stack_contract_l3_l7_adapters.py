# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (l3_l7_adapters) from former test_holonomic_adapter_stack_contract.py

from __future__ import annotations

from tests.holonomic_adapter_stack_contract_support import *  # noqa: F403

def test_l3_adapter_contract():
    params = L3_HolonomicParameters(n_genes=10)
    adapter = L3_GenomicAdapter(params)
    out = adapter.step_jax(0.1, inputs=jnp.ones((10, 1024)))
    assert out.shape == (10, 1024)
    metrics = adapter.get_metrics()
    assert "chromatin_coherence_r3" in metrics
    decoded = adapter.decode(out)
    assert "avg_accessibility" in decoded


def test_l4_adapter_contract():
    adapter = L4_CellularAdapter()
    out = adapter.step_jax(0.1)
    assert out.shape[0] == 400
    metrics = adapter.get_metrics()
    assert "order_parameter" in metrics


def test_l5_adapter_contract():
    adapter = L5_OrganismalAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "hrv_coherence_r5" in metrics


def test_l6_adapter_contract():
    adapter = L6_PlanetaryAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "gaia_potential" in metrics


def test_l6_percolation_threshold_controls_regional_coherence():
    low_threshold = L6_PlanetaryAdapter(
        L6_HolonomicParameters(n_regions=8, bitstream_length=16, p_percolation=0.2)
    )
    high_threshold = L6_PlanetaryAdapter(
        L6_HolonomicParameters(n_regions=8, bitstream_length=16, p_percolation=0.8)
    )
    inputs = jnp.full((8, 16), 0.5)

    low_threshold.step_jax(0.01, inputs=inputs)
    high_threshold.step_jax(0.01, inputs=inputs)

    assert float(jnp.mean(low_threshold.regional_coherence)) > float(
        jnp.mean(high_threshold.regional_coherence)
    )


def test_l6_quality_factor_amplifies_coherent_drive():
    low_q = L6_PlanetaryAdapter(
        L6_HolonomicParameters(n_regions=8, bitstream_length=16, q_factor=1.0)
    )
    high_q = L6_PlanetaryAdapter(
        L6_HolonomicParameters(n_regions=8, bitstream_length=16, q_factor=8.0)
    )
    inputs = jnp.ones((8, 16))

    low_q.step_jax(0.01, inputs=inputs)
    high_q.step_jax(0.01, inputs=inputs)

    assert float(jnp.mean(jnp.abs(high_q.phi_planetary))) > float(
        jnp.mean(jnp.abs(low_q.phi_planetary))
    )


def test_l6_rejects_invalid_holonomic_parameters():
    with pytest.raises(ValueError, match="n_regions"):
        L6_PlanetaryAdapter(L6_HolonomicParameters(n_regions=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L6_PlanetaryAdapter(L6_HolonomicParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="q_factor"):
        L6_PlanetaryAdapter(L6_HolonomicParameters(q_factor=0.0))
    with pytest.raises(ValueError, match="p_percolation"):
        L6_PlanetaryAdapter(L6_HolonomicParameters(p_percolation=1.0))


def test_l7_adapter_contract():
    adapter = L7_SymbolicAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((13, 1024)))
    assert out.shape == (13, 1024)
    metrics = adapter.get_metrics()
    assert "routing_coherence" in metrics


def test_l7_metatron_matrix_is_full_13_node_geometry():
    adapter = L7_SymbolicAdapter()
    matrix = np.asarray(adapter.metatron_matrix)

    assert matrix.shape == (13, 13)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(13), rtol=1e-7, atol=1e-7)
    assert np.all(np.diag(matrix) >= 0.05)
    assert np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 13 * 12
    assert matrix[0, 1] > matrix[0, 7]


def test_l7_rejects_invalid_routing_parameters():
    with pytest.raises(ValueError, match="n_nodes"):
        L7_SymbolicAdapter(L7_HolonomicParameters(n_nodes=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L7_SymbolicAdapter(L7_HolonomicParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="g_geometric_gain"):
        L7_SymbolicAdapter(L7_HolonomicParameters(g_geometric_gain=0.0))
    with pytest.raises(ValueError, match="coupling_leak"):
        L7_SymbolicAdapter(L7_HolonomicParameters(coupling_leak=1.0))


