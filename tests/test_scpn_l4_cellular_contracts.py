# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for SCPN L4 cellular layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l4_cellular import L4_CellularLayer, L4_StochasticParameters


def test_l4_seed_scopes_initial_state_and_output_bitstreams() -> None:
    params = L4_StochasticParameters(
        grid_size=(3, 3),
        bitstream_length=128,
        noise_amplitude=0.0,
        gap_junction_noise=0.0,
        rng_seed=123,
    )
    layer_a = L4_CellularLayer(params)
    layer_b = L4_CellularLayer(params)

    np.testing.assert_allclose(layer_a.phases, layer_b.phases)
    np.testing.assert_allclose(layer_a.calcium, layer_b.calcium)
    np.testing.assert_array_equal(layer_a.gap_junctions, layer_b.gap_junctions)
    out_a0 = layer_a.step(0.01)["output_bitstreams"]
    out_b0 = layer_b.step(0.01)["output_bitstreams"]
    out_a1 = layer_a.step(0.01)["output_bitstreams"]
    out_b1 = layer_b.step(0.01)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l4_gap_junction_conductance_and_organismal_drive_are_wired() -> None:
    common: dict[str, Any] = {
        "grid_size": (2, 2),
        "bitstream_length": 16,
        "noise_amplitude": 0.0,
        "gap_junction_noise": 0.0,
        "ca_decay_rate": 0.0,
        "organismal_coupling": 0.25,
        "rng_seed": 456,
    }
    blocked = L4_CellularLayer(L4_StochasticParameters(**common, gap_junction_conductance=0.0))
    coupled = L4_CellularLayer(L4_StochasticParameters(**common, gap_junction_conductance=1.0))
    for layer in (blocked, coupled):
        layer.calcium = np.array([1.0, 0.0, 0.0, 0.0])
        layer.gap_junctions = np.ones(layer.n_cells)

    blocked_result = blocked.step(0.01)
    coupled_result = coupled.step(0.01)

    assert coupled_result["calcium"][1] > blocked_result["calcium"][1]
    assert coupled_result["organismal_drive"] == pytest.approx(
        coupled.params.organismal_coupling * coupled_result["synchronization"]
    )


def test_l4_genomic_coupling_uses_validated_protein_levels() -> None:
    layer = L4_CellularLayer(
        L4_StochasticParameters(
            grid_size=(2, 2),
            bitstream_length=16,
            noise_amplitude=0.0,
            gap_junction_noise=0.0,
            genomic_coupling=0.5,
        )
    )

    before = layer.amplitudes.copy()
    result = layer.step(0.01, l3_input={"protein_levels": np.ones(4)})

    assert np.mean(result["amplitudes"]) > np.mean(before)


def test_l4_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="grid_size"):
        L4_CellularLayer(L4_StochasticParameters(grid_size=(0, 2)))
    with pytest.raises(ValueError, match="bitstream_length"):
        L4_CellularLayer(L4_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="natural_frequency"):
        L4_CellularLayer(L4_StochasticParameters(natural_frequency=0.0))
    with pytest.raises(ValueError, match="coupling_strength"):
        L4_CellularLayer(L4_StochasticParameters(coupling_strength=-0.1))
    with pytest.raises(ValueError, match="noise_amplitude"):
        L4_CellularLayer(L4_StochasticParameters(noise_amplitude=-0.1))
    with pytest.raises(ValueError, match="gap_junction_conductance"):
        L4_CellularLayer(L4_StochasticParameters(gap_junction_conductance=1.1))
    with pytest.raises(ValueError, match="gap_junction_noise"):
        L4_CellularLayer(L4_StochasticParameters(gap_junction_noise=-0.1))
    with pytest.raises(ValueError, match="ca_diffusion_rate"):
        L4_CellularLayer(L4_StochasticParameters(ca_diffusion_rate=-0.1))
    with pytest.raises(ValueError, match="ca_decay_rate"):
        L4_CellularLayer(L4_StochasticParameters(ca_decay_rate=-0.1))
    with pytest.raises(ValueError, match="ca_release_threshold"):
        L4_CellularLayer(L4_StochasticParameters(ca_release_threshold=1.1))
    with pytest.raises(ValueError, match="genomic_coupling"):
        L4_CellularLayer(L4_StochasticParameters(genomic_coupling=-0.1))
    with pytest.raises(ValueError, match="organismal_coupling"):
        L4_CellularLayer(L4_StochasticParameters(organismal_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L4_CellularLayer(L4_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L4_CellularLayer(L4_StochasticParameters(grid_size=(2, 2), bitstream_length=16))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="protein_levels"):
        layer.step(0.01, l3_input={"protein_levels": np.array([1.0, np.nan])})
    with pytest.raises(ValueError, match="external_stimulus"):
        layer.step(0.01, external_stimulus=np.ones(3))
    with pytest.raises(ValueError, match="external_stimulus"):
        layer.step(0.01, external_stimulus=np.array([0.0, 0.0, 0.0, np.nan]))


def test_l4_valid_external_stimulus_and_state_accessors() -> None:
    layer = L4_CellularLayer(
        L4_StochasticParameters(grid_size=(2, 2), bitstream_length=16, rng_seed=5)
    )
    # A valid full-grid stimulus is applied to the calcium field.
    result = layer.step(0.01, external_stimulus=np.ones(4, dtype=np.float64))
    assert "output_bitstreams" in result

    metric = layer.get_global_metric()
    assert 0.0 <= metric <= 1.0
    assert layer.get_tissue_pattern().shape == (2, 2)


def test_l4_negative_seed_rejected() -> None:
    with pytest.raises(ValueError, match="rng_seed"):
        L4_CellularLayer(L4_StochasticParameters(rng_seed=-1))
