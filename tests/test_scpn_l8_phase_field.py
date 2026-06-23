# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SCPN L8 phase-field layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l8_phase_field import (
    L8_PhaseFieldLayer,
    L8_StochasticParameters,
)


def test_l8_default_pulsar_frequencies_match_configured_size() -> None:
    layer = L8_PhaseFieldLayer(
        L8_StochasticParameters(n_pulsars=3, bitstream_length=16, rng_seed=8)
    )

    assert layer.params.pulsar_omegas is not None
    assert layer.params.pulsar_omegas.shape == (3,)
    result = layer.step(0.001)
    assert result["output_bitstreams"].shape == (3, 16)


def test_l8_symbolic_coupling_controls_phase_drive() -> None:
    common = dict(
        n_pulsars=2,
        bitstream_length=16,
        k_cosmic=0.0,
        director_coupling=0.0,
        pulsar_omegas=np.zeros(2, dtype=np.float64),
        rng_seed=18,
    )
    uncoupled = L8_PhaseFieldLayer(L8_StochasticParameters(**common, symbolic_coupling=0.0))
    coupled = L8_PhaseFieldLayer(L8_StochasticParameters(**common, symbolic_coupling=0.5))
    uncoupled.phases = np.full(2, np.pi / 2.0, dtype=np.float64)
    coupled.phases = np.full(2, np.pi / 2.0, dtype=np.float64)

    base = uncoupled.step(0.2, {"glyph_vector": np.ones(2, dtype=np.float64)})["phases"]
    driven = coupled.step(0.2, {"glyph_vector": np.ones(2, dtype=np.float64)})["phases"]

    np.testing.assert_allclose(base, np.full(2, np.pi / 2.0))
    np.testing.assert_allclose(driven, np.full(2, np.pi / 2.0 - 0.1))


def test_l8_consumes_l7_cosmic_phase_drive_contract() -> None:
    common = dict(
        n_pulsars=2,
        bitstream_length=16,
        k_cosmic=0.0,
        director_coupling=0.0,
        pulsar_omegas=np.zeros(2, dtype=np.float64),
        rng_seed=19,
    )
    base_layer = L8_PhaseFieldLayer(L8_StochasticParameters(**common, symbolic_coupling=0.5))
    driven_layer = L8_PhaseFieldLayer(L8_StochasticParameters(**common, symbolic_coupling=0.5))
    base_layer.phases = np.full(2, np.pi / 2.0, dtype=np.float64)
    driven_layer.phases = np.full(2, np.pi / 2.0, dtype=np.float64)

    base = base_layer.step(0.2)["phases"]
    driven = driven_layer.step(0.2, {"cosmic_phase_drive": 1.0})["phases"]

    np.testing.assert_allclose(base, np.full(2, np.pi / 2.0))
    np.testing.assert_allclose(driven, np.full(2, np.pi / 2.0 - 0.1))


def test_l8_prefers_structured_cosmic_phase_drive_over_glyph_fallback() -> None:
    params = L8_StochasticParameters(
        n_pulsars=2,
        bitstream_length=16,
        k_cosmic=0.0,
        symbolic_coupling=0.5,
        director_coupling=0.0,
        pulsar_omegas=np.zeros(2, dtype=np.float64),
        rng_seed=20,
    )
    drive_only = L8_PhaseFieldLayer(params)
    both_payloads = L8_PhaseFieldLayer(params)
    drive_only.phases = np.full(2, np.pi / 2.0, dtype=np.float64)
    both_payloads.phases = np.full(2, np.pi / 2.0, dtype=np.float64)

    drive_only_phases = drive_only.step(0.2, {"cosmic_phase_drive": 1.0})["phases"]
    both_phases = both_payloads.step(
        0.2,
        {
            "glyph_vector": np.zeros(2, dtype=np.float64),
            "cosmic_phase_drive": 1.0,
        },
    )["phases"]

    np.testing.assert_allclose(both_phases, drive_only_phases)


def test_l8_seed_scopes_initial_phases_and_output_bitstreams() -> None:
    params = L8_StochasticParameters(
        n_pulsars=3,
        bitstream_length=64,
        k_cosmic=0.0,
        pulsar_omegas=np.zeros(3, dtype=np.float64),
        rng_seed=123,
    )
    layer_a = L8_PhaseFieldLayer(params)
    layer_b = L8_PhaseFieldLayer(params)

    np.testing.assert_allclose(layer_a.phases, layer_b.phases)
    out_a0 = layer_a.step(0.001)["output_bitstreams"]
    out_b0 = layer_b.step(0.001)["output_bitstreams"]
    out_a1 = layer_a.step(0.001)["output_bitstreams"]
    out_b1 = layer_b.step(0.001)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l8_director_coupling_is_exposed_as_downstream_drive() -> None:
    layer = L8_PhaseFieldLayer(
        L8_StochasticParameters(
            n_pulsars=2,
            bitstream_length=16,
            director_coupling=0.25,
            rng_seed=42,
        )
    )

    result = layer.step(0.001)

    assert result["director_drive"] == pytest.approx(0.25 * result["cosmic_alignment"])


def test_l8_exports_memory_imprint_reference_wave_for_l9() -> None:
    layer = L8_PhaseFieldLayer(
        L8_StochasticParameters(
            n_pulsars=2,
            bitstream_length=16,
            k_cosmic=0.0,
            symbolic_coupling=0.0,
            director_coupling=0.0,
            pulsar_omegas=np.zeros(2, dtype=np.float64),
            rng_seed=81,
        )
    )
    layer.phases = np.array([0.0, np.pi / 2.0], dtype=np.float64)

    result = layer.step(0.001)
    reference = result["memory_imprint_drive"]

    expected_order = np.mean(np.exp(1j * result["phases"]))
    expected_amplitude = float(np.abs(expected_order))
    expected_phase = float(np.angle(expected_order))
    assert reference["reference_amplitude"] == pytest.approx(expected_amplitude)
    assert reference["reference_phase"] == pytest.approx(expected_phase)
    assert reference["reference_real"] == pytest.approx(expected_amplitude * np.cos(expected_phase))


def test_l8_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_pulsars"):
        L8_PhaseFieldLayer(L8_StochasticParameters(n_pulsars=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L8_PhaseFieldLayer(L8_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="k_cosmic"):
        L8_PhaseFieldLayer(L8_StochasticParameters(k_cosmic=-0.1))
    with pytest.raises(ValueError, match="symbolic_coupling"):
        L8_PhaseFieldLayer(L8_StochasticParameters(symbolic_coupling=-0.1))
    with pytest.raises(ValueError, match="director_coupling"):
        L8_PhaseFieldLayer(L8_StochasticParameters(director_coupling=-0.1))
    with pytest.raises(ValueError, match="pulsar_omegas"):
        L8_PhaseFieldLayer(
            L8_StochasticParameters(n_pulsars=2, pulsar_omegas=np.array([1.0, np.nan]))
        )
    with pytest.raises(ValueError, match="pulsar_omegas"):
        L8_PhaseFieldLayer(L8_StochasticParameters(n_pulsars=2, pulsar_omegas=np.array([1.0])))
    with pytest.raises(ValueError, match="rng_seed"):
        L8_PhaseFieldLayer(L8_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L8_PhaseFieldLayer(L8_StochasticParameters(n_pulsars=2, rng_seed=5))
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="glyph_vector"):
        layer.step(0.001, {"glyph_vector": np.array([1.0, np.nan])})
    with pytest.raises(ValueError, match="glyph_vector"):
        layer.step(0.001, {"glyph_vector": np.array([])})
    with pytest.raises(ValueError, match="cosmic_phase_drive"):
        layer.step(0.001, {"cosmic_phase_drive": np.nan})
    with pytest.raises(ValueError, match="cosmic_phase_drive"):
        layer.step(0.001, {"cosmic_phase_drive": np.array([0.1, 0.2])})
    with pytest.raises(ValueError, match="cosmic_phase_drive"):
        layer.step(0.001, {"cosmic_phase_drive": -0.1})


def test_l8_get_global_metric_order_parameter() -> None:
    layer = L8_PhaseFieldLayer(
        L8_StochasticParameters(n_pulsars=4, bitstream_length=16, rng_seed=2)
    )
    assert 0.0 <= layer.get_global_metric() <= 1.0


def test_l8_validate_params_type_guards_omegas_and_negative_seed() -> None:
    # __post_init__ rejects a bool n_pulsars during construction (np.resize), so
    # exercise the validator directly with fields mutated after a valid build.
    def _params() -> L8_StochasticParameters:
        return L8_StochasticParameters(n_pulsars=3, bitstream_length=16, rng_seed=1)

    bad_pulsars = _params()
    bad_pulsars.n_pulsars = cast(int, True)
    with pytest.raises(ValueError, match="n_pulsars must be a positive integer"):
        L8_PhaseFieldLayer._validate_params(bad_pulsars)

    bad_bits = _params()
    bad_bits.bitstream_length = cast(int, True)
    with pytest.raises(ValueError, match="bitstream_length must be a positive integer"):
        L8_PhaseFieldLayer._validate_params(bad_bits)

    no_omegas = _params()
    no_omegas.pulsar_omegas = None
    with pytest.raises(ValueError, match="pulsar_omegas must be initialised"):
        L8_PhaseFieldLayer._validate_params(no_omegas)

    bad_seed = _params()
    bad_seed.rng_seed = -1
    with pytest.raises(ValueError, match="rng_seed"):
        L8_PhaseFieldLayer._validate_params(bad_seed)


def test_l8_l7_phase_drive_is_neutral_without_known_keys() -> None:
    assert L8_PhaseFieldLayer._l7_phase_drive({"unrelated_channel": 1.0}) == 0.0
