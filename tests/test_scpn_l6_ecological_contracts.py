# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for SCPN L6 ecological layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l6_ecological import L6_EcologicalLayer, L6_StochasticParameters


def test_l6_seed_scopes_initial_state_and_output_bitstreams() -> None:
    params = L6_StochasticParameters(
        n_field_nodes=12,
        bitstream_length=128,
        schumann_noise=0.0,
        geomag_variation=0.0,
        network_noise=0.0,
        rng_seed=123,
    )
    layer_a = L6_EcologicalLayer(params)
    layer_b = L6_EcologicalLayer(params)

    np.testing.assert_allclose(layer_a.biospheric_field, layer_b.biospheric_field)
    out_a0 = layer_a.step(0.01, solar_activity=0.4, lunar_phase=0.2)["output_bitstreams"]
    out_b0 = layer_b.step(0.01, solar_activity=0.4, lunar_phase=0.2)["output_bitstreams"]
    out_a1 = layer_a.step(0.01, solar_activity=0.4, lunar_phase=0.2)["output_bitstreams"]
    out_b1 = layer_b.step(0.01, solar_activity=0.4, lunar_phase=0.2)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l6_symbolic_coupling_exports_l7_drive() -> None:
    params = L6_StochasticParameters(
        n_field_nodes=16,
        bitstream_length=16,
        schumann_noise=0.0,
        geomag_variation=0.0,
        network_noise=0.0,
        symbolic_coupling=0.25,
        rng_seed=456,
    )
    layer = L6_EcologicalLayer(params)

    result = layer.step(0.01, solar_activity=0.5, lunar_phase=0.0)

    assert result["symbolic_drive"].shape == (params.n_field_nodes,)
    np.testing.assert_allclose(
        result["symbolic_drive"], params.symbolic_coupling * result["schumann_field"]
    )


def test_l6_organismal_coupling_uses_validated_emotional_state() -> None:
    base = L6_EcologicalLayer(
        L6_StochasticParameters(
            n_field_nodes=16,
            bitstream_length=16,
            schumann_noise=0.0,
            geomag_variation=0.0,
            network_noise=0.0,
            rng_seed=789,
        )
    )
    coupled = L6_EcologicalLayer(base.params)

    without_l5 = base.step(0.01)["biospheric_field"]
    with_l5 = coupled.step(0.01, l5_input={"emotional_state": np.ones(8)})["biospheric_field"]

    assert np.mean(with_l5) > np.mean(without_l5)


def test_l6_consumes_l5_ecological_drive_contract() -> None:
    base = L6_EcologicalLayer(
        L6_StochasticParameters(
            n_field_nodes=16,
            bitstream_length=16,
            schumann_noise=0.0,
            geomag_variation=0.0,
            network_noise=0.0,
            organismal_coupling=1.0,
            rng_seed=790,
        )
    )
    driven = L6_EcologicalLayer(base.params)

    without_drive = base.step(0.01)["biospheric_field"]
    with_drive = driven.step(0.01, l5_input={"ecological_drive": np.ones(8)})["biospheric_field"]

    assert np.mean(with_drive) > np.mean(without_drive)


def test_l6_prefers_structured_ecological_drive_over_fallback_emotional_mean() -> None:
    params = L6_StochasticParameters(
        n_field_nodes=16,
        bitstream_length=16,
        schumann_noise=0.0,
        geomag_variation=0.0,
        network_noise=0.0,
        organismal_coupling=1.0,
        rng_seed=791,
    )
    drive_only = L6_EcologicalLayer(params)
    both_payloads = L6_EcologicalLayer(params)

    drive_only_field = drive_only.step(0.01, l5_input={"ecological_drive": np.ones(8)})[
        "biospheric_field"
    ]
    both_field = both_payloads.step(
        0.01,
        l5_input={
            "emotional_state": np.zeros(8),
            "ecological_drive": np.ones(8),
        },
    )["biospheric_field"]

    np.testing.assert_allclose(both_field, drive_only_field)


def test_l6_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_field_nodes"):
        L6_EcologicalLayer(L6_StochasticParameters(n_field_nodes=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L6_EcologicalLayer(L6_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="schumann_frequencies"):
        L6_EcologicalLayer(L6_StochasticParameters(schumann_frequencies=()))
    with pytest.raises(ValueError, match="schumann_frequencies"):
        L6_EcologicalLayer(L6_StochasticParameters(schumann_frequencies=(7.83, np.nan)))
    with pytest.raises(ValueError, match="schumann_amplitude"):
        L6_EcologicalLayer(L6_StochasticParameters(schumann_amplitude=-0.1))
    with pytest.raises(ValueError, match="schumann_noise"):
        L6_EcologicalLayer(L6_StochasticParameters(schumann_noise=-0.1))
    with pytest.raises(ValueError, match="geomag_baseline"):
        L6_EcologicalLayer(L6_StochasticParameters(geomag_baseline=0.0))
    with pytest.raises(ValueError, match="geomag_variation"):
        L6_EcologicalLayer(L6_StochasticParameters(geomag_variation=-0.1))
    with pytest.raises(ValueError, match="circadian_period"):
        L6_EcologicalLayer(L6_StochasticParameters(circadian_period=0.0))
    with pytest.raises(ValueError, match="circadian_amplitude"):
        L6_EcologicalLayer(L6_StochasticParameters(circadian_amplitude=-0.1))
    with pytest.raises(ValueError, match="network_coupling"):
        L6_EcologicalLayer(L6_StochasticParameters(network_coupling=-0.1))
    with pytest.raises(ValueError, match="network_noise"):
        L6_EcologicalLayer(L6_StochasticParameters(network_noise=-0.1))
    with pytest.raises(ValueError, match="organismal_coupling"):
        L6_EcologicalLayer(L6_StochasticParameters(organismal_coupling=-0.1))
    with pytest.raises(ValueError, match="symbolic_coupling"):
        L6_EcologicalLayer(L6_StochasticParameters(symbolic_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L6_EcologicalLayer(L6_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L6_EcologicalLayer(L6_StochasticParameters(n_field_nodes=8, bitstream_length=16))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="solar_activity"):
        layer.step(0.01, solar_activity=np.nan)
    with pytest.raises(ValueError, match="solar_activity"):
        layer.step(0.01, solar_activity=1.1)
    with pytest.raises(ValueError, match="lunar_phase"):
        layer.step(0.01, lunar_phase=np.nan)
    with pytest.raises(ValueError, match="emotional_state"):
        layer.step(0.01, l5_input={"emotional_state": np.array([0.5, np.nan])})
    with pytest.raises(ValueError, match="ecological_drive"):
        layer.step(0.01, l5_input={"ecological_drive": np.array([0.5, np.nan])})
    with pytest.raises(ValueError, match="ecological_drive"):
        layer.step(0.01, l5_input={"ecological_drive": np.array([-0.1, 0.2])})


def test_l6_history_trim_and_state_accessors() -> None:
    layer = L6_EcologicalLayer(L6_StochasticParameters(rng_seed=4))
    for _ in range(110):
        layer.step(0.01, solar_activity=0.4, lunar_phase=0.2)
    assert len(layer.history) <= 100  # rolling 100-step window
    assert isinstance(layer.get_global_metric(), float)
    assert isinstance(layer.get_schumann_spectrum(), dict)
    assert 0.0 <= layer.get_circadian_time() <= 24.0


def test_l6_circadian_amplitude_bound_and_neutral_l5_payload() -> None:
    with pytest.raises(ValueError, match="circadian_amplitude"):
        L6_EcologicalLayer(L6_StochasticParameters(circadian_amplitude=0.9))
    # An l5 payload with neither known key contributes no organismal effect.
    layer = L6_EcologicalLayer(L6_StochasticParameters(rng_seed=5))
    result = layer.step(0.01, solar_activity=0.5, lunar_phase=0.0, l5_input={"unrelated": 1.0})
    assert "output_bitstreams" in result
