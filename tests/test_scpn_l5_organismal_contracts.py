# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for SCPN L5 organismal layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l5_organismal import L5_OrganismalLayer, L5_StochasticParameters


def test_l5_seed_scopes_initial_state_and_output_bitstreams() -> None:
    params = L5_StochasticParameters(
        n_autonomic_nodes=16,
        bitstream_length=128,
        emotional_noise=0.0,
        rng_seed=123,
    )
    layer_a = L5_OrganismalLayer(params)
    layer_b = L5_OrganismalLayer(params)

    np.testing.assert_allclose(layer_a.interoceptive_state, layer_b.interoceptive_state)
    out_a0 = layer_a.step(0.01)["output_bitstreams"]
    out_b0 = layer_b.step(0.01)["output_bitstreams"]
    out_a1 = layer_a.step(0.01)["output_bitstreams"]
    out_b1 = layer_b.step(0.01)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l5_stress_event_is_validated_and_applied() -> None:
    layer = L5_OrganismalLayer(
        L5_StochasticParameters(n_autonomic_nodes=16, bitstream_length=16, emotional_noise=0.0)
    )

    before = layer.emotional_state.copy()
    result = layer.step(0.01, external_event={"type": "stress", "intensity": 0.8})

    assert result["emotional_state"][layer.AROUSAL] > before[layer.AROUSAL]
    assert result["emotional_state"][layer.SAFETY] < before[layer.SAFETY]


def test_l5_cellular_and_ecological_couplings_are_validated_and_exported() -> None:
    params = L5_StochasticParameters(
        n_autonomic_nodes=16,
        bitstream_length=16,
        emotional_noise=0.0,
        cellular_coupling=0.25,
        ecological_coupling=0.5,
        rng_seed=456,
    )
    layer = L5_OrganismalLayer(params)

    result = layer.step(0.01, l4_input={"synchronization": 1.0})

    assert result["emotional_state"][layer.CERTAINTY] > 0.5
    assert result["ecological_drive"].shape == (params.n_emotional_dims,)
    np.testing.assert_allclose(
        result["ecological_drive"], params.ecological_coupling * result["emotional_state"]
    )


def test_l5_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_emotional_dims"):
        L5_OrganismalLayer(L5_StochasticParameters(n_emotional_dims=7))
    with pytest.raises(ValueError, match="n_autonomic_nodes"):
        L5_OrganismalLayer(L5_StochasticParameters(n_autonomic_nodes=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L5_OrganismalLayer(L5_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="sympathetic_baseline"):
        L5_OrganismalLayer(L5_StochasticParameters(sympathetic_baseline=np.nan))
    with pytest.raises(ValueError, match="parasympathetic_baseline"):
        L5_OrganismalLayer(L5_StochasticParameters(parasympathetic_baseline=1.1))
    with pytest.raises(ValueError, match="autonomic_time_constant"):
        L5_OrganismalLayer(L5_StochasticParameters(autonomic_time_constant=0.0))
    with pytest.raises(ValueError, match="base_heart_rate"):
        L5_OrganismalLayer(L5_StochasticParameters(base_heart_rate=0.0))
    with pytest.raises(ValueError, match="hrv_amplitude"):
        L5_OrganismalLayer(L5_StochasticParameters(hrv_amplitude=-0.1))
    with pytest.raises(ValueError, match="respiratory_frequency"):
        L5_OrganismalLayer(L5_StochasticParameters(respiratory_frequency=0.0))
    with pytest.raises(ValueError, match="emotional_decay"):
        L5_OrganismalLayer(L5_StochasticParameters(emotional_decay=-0.1))
    with pytest.raises(ValueError, match="emotional_noise"):
        L5_OrganismalLayer(L5_StochasticParameters(emotional_noise=-0.1))
    with pytest.raises(ValueError, match="attractor_strength"):
        L5_OrganismalLayer(L5_StochasticParameters(attractor_strength=-0.1))
    with pytest.raises(ValueError, match="cellular_coupling"):
        L5_OrganismalLayer(L5_StochasticParameters(cellular_coupling=-0.1))
    with pytest.raises(ValueError, match="ecological_coupling"):
        L5_OrganismalLayer(L5_StochasticParameters(ecological_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L5_OrganismalLayer(L5_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L5_OrganismalLayer(L5_StochasticParameters(n_autonomic_nodes=8, bitstream_length=16))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="synchronization"):
        layer.step(0.01, l4_input={"synchronization": np.nan})
    with pytest.raises(ValueError, match="external_event"):
        layer.step(0.01, external_event=cast(Any, {0: np.nan}))
    with pytest.raises(ValueError, match="intensity"):
        layer.step(0.01, external_event={"type": "stress", "intensity": np.nan})


def test_l5_rejects_negative_rng_seed() -> None:
    with pytest.raises(ValueError, match="rng_seed"):
        L5_OrganismalLayer(L5_StochasticParameters(rng_seed=-1))


def test_l5_rejects_unknown_external_event_type_and_keys() -> None:
    layer = L5_OrganismalLayer(L5_StochasticParameters(n_autonomic_nodes=8, bitstream_length=16))
    with pytest.raises(ValueError, match="external_event type"):
        layer.step(0.01, external_event={"type": "euphoria"})
    with pytest.raises(ValueError, match="dimension values must be finite"):
        layer.step(0.01, external_event={"valence": np.nan})
    with pytest.raises(ValueError, match="external_event keys"):
        layer.step(0.01, external_event=cast(Any, {"banana": 0.5}))


def test_l5_trims_rr_interval_history_beyond_window() -> None:
    layer = L5_OrganismalLayer(
        L5_StochasticParameters(
            n_autonomic_nodes=6, bitstream_length=8, emotional_noise=0.0, rng_seed=1
        )
    )
    for _ in range(150):
        layer.step(0.01)
    assert len(layer.rr_intervals) == 100


def test_l5_get_global_metric_combines_hrv_and_emotional_stability() -> None:
    layer = L5_OrganismalLayer(
        L5_StochasticParameters(
            n_autonomic_nodes=8, bitstream_length=16, emotional_noise=0.0, rng_seed=7
        )
    )
    for _ in range(5):
        layer.step(0.01)

    metric = layer.get_global_metric()
    assert isinstance(metric, float)
    assert 0.0 <= metric <= 1.0
    assert layer.get_emotional_valence() == pytest.approx(
        float(layer.emotional_state[layer.VALENCE])
    )


def test_l5_calm_and_reward_events_shift_emotional_state() -> None:
    calm_layer = L5_OrganismalLayer(
        L5_StochasticParameters(
            n_autonomic_nodes=8, bitstream_length=16, emotional_noise=0.0, rng_seed=3
        )
    )
    calm_before = calm_layer.emotional_state.copy()
    calm = calm_layer.step(0.01, external_event={"type": "calm", "intensity": 0.8})
    assert calm["emotional_state"][calm_layer.SAFETY] > calm_before[calm_layer.SAFETY]
    assert calm["emotional_state"][calm_layer.AROUSAL] < calm_before[calm_layer.AROUSAL]

    reward_layer = L5_OrganismalLayer(
        L5_StochasticParameters(
            n_autonomic_nodes=8, bitstream_length=16, emotional_noise=0.0, rng_seed=3
        )
    )
    reward_before = reward_layer.emotional_state.copy()
    reward = reward_layer.step(0.01, external_event={"type": "reward", "intensity": 0.8})
    assert reward["emotional_state"][reward_layer.VALENCE] > reward_before[reward_layer.VALENCE]
    assert reward["emotional_state"][reward_layer.APPROACH] > reward_before[reward_layer.APPROACH]


def test_l5_named_dimension_event_key_adjusts_that_dimension() -> None:
    layer = L5_OrganismalLayer(
        L5_StochasticParameters(
            n_autonomic_nodes=8, bitstream_length=16, emotional_noise=0.0, rng_seed=9
        )
    )
    before = layer.emotional_state.copy()
    result = layer.step(0.01, external_event={"fairness": 0.5})
    assert result["emotional_state"][layer.FAIRNESS] > before[layer.FAIRNESS]


def test_l5_integer_dimension_event_key_adjusts_that_dimension() -> None:
    layer = L5_OrganismalLayer(
        L5_StochasticParameters(
            n_autonomic_nodes=8, bitstream_length=16, emotional_noise=0.0, rng_seed=11
        )
    )
    before = layer.emotional_state.copy()
    result = layer.step(0.01, external_event=cast(Any, {layer.DOMINANCE: 0.4}))
    assert result["emotional_state"][layer.DOMINANCE] > before[layer.DOMINANCE]
