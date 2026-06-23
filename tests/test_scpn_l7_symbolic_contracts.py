# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contract tests for SCPN L7 symbolic layer

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l7_symbolic import L7_StochasticParameters, L7_SymbolicLayer


def test_l7_seed_scopes_initial_state_and_output_bitstreams() -> None:
    params = L7_StochasticParameters(
        n_symbols=16,
        n_meridians=4,
        n_acupoints=16,
        bitstream_length=64,
        rng_seed=123,
    )
    layer_a = L7_SymbolicLayer(params)
    layer_b = L7_SymbolicLayer(params)

    np.testing.assert_allclose(layer_a.symbol_activations, layer_b.symbol_activations)
    np.testing.assert_allclose(layer_a.e8_state, layer_b.e8_state)
    out_a0 = layer_a.step(0.001)["output_bitstreams"]
    out_b0 = layer_b.step(0.001)["output_bitstreams"]
    out_a1 = layer_a.step(0.001)["output_bitstreams"]
    out_b1 = layer_b.step(0.001)["output_bitstreams"]

    np.testing.assert_array_equal(out_a0, out_b0)
    np.testing.assert_array_equal(out_a1, out_b1)
    assert not np.array_equal(out_a0, out_a1)


def test_l7_e8_alignment_uses_full_root_system() -> None:
    roots = L7_SymbolicLayer._e8_roots()
    assert roots.shape == (240, 8)

    layer = L7_SymbolicLayer(
        L7_StochasticParameters(
            n_symbols=16,
            n_meridians=4,
            n_acupoints=16,
            bitstream_length=16,
            rng_seed=7,
        )
    )
    layer.e8_state = np.full(8, 0.5, dtype=np.float64)

    result = layer.step(0.001)

    assert result["e8_alignment"] == pytest.approx(1.0)


def test_l7_symbolic_health_weights_are_validated_and_used() -> None:
    params = L7_StochasticParameters(
        n_symbols=16,
        n_meridians=4,
        n_acupoints=16,
        bitstream_length=16,
        phi_alignment_weight=1.0,
        fibonacci_weight=0.0,
        metatron_weight=0.0,
        platonic_weight=0.0,
        e8_weight=0.0,
        rng_seed=8,
    )
    layer = L7_SymbolicLayer(params)
    result = layer.step(0.001)

    assert result["symbolic_health"] == pytest.approx(result["phi_alignment"])


def test_l7_cosmic_coupling_exports_l8_phase_drive() -> None:
    params = L7_StochasticParameters(
        n_symbols=16,
        n_meridians=4,
        n_acupoints=16,
        bitstream_length=16,
        cosmic_coupling=0.5,
        rng_seed=12,
    )
    layer = L7_SymbolicLayer(params)

    result = layer.step(0.001)

    assert result["cosmic_phase_drive"] == pytest.approx(
        params.cosmic_coupling * result["symbolic_health"]
    )


def test_l7_consumes_l6_symbolic_drive_contract() -> None:
    params = L7_StochasticParameters(
        n_symbols=16,
        n_meridians=4,
        n_acupoints=16,
        bitstream_length=16,
        ecological_coupling=0.2,
        rng_seed=10,
    )
    base = L7_SymbolicLayer(params)
    driven = L7_SymbolicLayer(params)

    base_qi = base.step(0.001)["meridian_qi"]
    driven_qi = driven.step(0.001, l6_input={"symbolic_drive": np.ones(8)})["meridian_qi"]

    assert np.mean(driven_qi) > np.mean(base_qi)


def test_l7_prefers_structured_symbolic_drive_over_schumann_fallback() -> None:
    params = L7_StochasticParameters(
        n_symbols=16,
        n_meridians=4,
        n_acupoints=16,
        bitstream_length=16,
        ecological_coupling=0.2,
        rng_seed=11,
    )
    drive_only = L7_SymbolicLayer(params)
    both_payloads = L7_SymbolicLayer(params)

    drive_only_qi = drive_only.step(0.001, l6_input={"symbolic_drive": np.ones(8)})["meridian_qi"]
    both_qi = both_payloads.step(
        0.001,
        l6_input={
            "schumann_field": np.zeros(8),
            "symbolic_drive": np.ones(8),
        },
    )["meridian_qi"]

    np.testing.assert_allclose(both_qi, drive_only_qi)


def test_l7_rejects_invalid_parameters_and_inputs() -> None:
    with pytest.raises(ValueError, match="n_symbols"):
        L7_SymbolicLayer(L7_StochasticParameters(n_symbols=1))
    with pytest.raises(ValueError, match="n_meridians"):
        L7_SymbolicLayer(L7_StochasticParameters(n_meridians=0))
    with pytest.raises(ValueError, match="n_acupoints"):
        L7_SymbolicLayer(L7_StochasticParameters(n_acupoints=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L7_SymbolicLayer(L7_StochasticParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="glyph_dimensions"):
        L7_SymbolicLayer(L7_StochasticParameters(glyph_dimensions=5))
    with pytest.raises(ValueError, match="weights"):
        L7_SymbolicLayer(L7_StochasticParameters(phi_alignment_weight=np.nan))
    with pytest.raises(ValueError, match="symbol_decay"):
        L7_SymbolicLayer(L7_StochasticParameters(symbol_decay=-0.1))
    with pytest.raises(ValueError, match="symbol_coupling"):
        L7_SymbolicLayer(L7_StochasticParameters(symbol_coupling=-0.1))
    with pytest.raises(ValueError, match="ecological_coupling"):
        L7_SymbolicLayer(L7_StochasticParameters(ecological_coupling=-0.1))
    with pytest.raises(ValueError, match="cosmic_coupling"):
        L7_SymbolicLayer(L7_StochasticParameters(cosmic_coupling=-0.1))
    with pytest.raises(ValueError, match="rng_seed"):
        L7_SymbolicLayer(L7_StochasticParameters(rng_seed=cast(Any, 1.5)))

    layer = L7_SymbolicLayer(
        L7_StochasticParameters(n_symbols=16, n_meridians=4, n_acupoints=16, rng_seed=9)
    )
    with pytest.raises(ValueError, match="dt"):
        layer.step(0.0)
    with pytest.raises(ValueError, match="symbol_input"):
        layer.step(0.001, symbol_input=np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="symbol_input"):
        layer.step(0.001, symbol_input=np.ones(15, dtype=np.float64))
    with pytest.raises(ValueError, match="schumann_field"):
        layer.step(0.001, l6_input={"schumann_field": np.array([0.5, np.nan])})
    with pytest.raises(ValueError, match="symbolic_drive"):
        layer.step(0.001, l6_input={"symbolic_drive": np.array([0.5, np.nan])})
    with pytest.raises(ValueError, match="symbolic_drive"):
        layer.step(0.001, l6_input={"symbolic_drive": np.array([-0.1, 0.2])})
    with pytest.raises(ValueError, match="acupoint_stimulus"):
        layer.step(0.001, acupoint_stimulus={0: np.nan})
    with pytest.raises(ValueError, match="acupoint_stimulus"):
        layer.step(0.001, acupoint_stimulus={16: 0.5})


def _layer(**overrides: Any) -> L7_SymbolicLayer:
    params = dict(n_symbols=16, n_meridians=4, n_acupoints=16, bitstream_length=16, rng_seed=5)
    params.update(overrides)
    return L7_SymbolicLayer(L7_StochasticParameters(**params))


def test_l7_step_consumes_valid_symbol_and_acupoint_inputs() -> None:
    layer = _layer()
    result = layer.step(
        0.001,
        symbol_input=np.ones(16, dtype=np.float64),
        acupoint_stimulus={0: 0.5, 3: 0.25},
    )
    assert "glyph_vector" in result
    assert layer.acupoint_activations[0] > 0.0
    assert layer.acupoint_activations[3] > 0.0
    # get_global_metric mirrors the assembled symbolic health.
    assert layer.get_global_metric() == layer.symbolic_health


def test_l7_neutral_alignments_with_silent_state() -> None:
    layer = _layer()
    layer.symbol_activations = np.zeros_like(layer.symbol_activations)
    layer.e8_state = np.zeros(8, dtype=np.float64)
    result = layer.step(0.001)
    # Sub-threshold activations and a zero E8 state take the neutral fallbacks.
    assert result["phi_alignment"] == 0.5
    assert result["fibonacci_alignment"] == 0.5
    assert result["e8_alignment"] == 0.5


def test_l7_l6_schumann_fallback_and_neutral_payload() -> None:
    # schumann_field (no symbolic_drive) drives the finite-mean fallback branch.
    schumann = _layer(ecological_coupling=0.2)
    schumann.step(0.001, l6_input={"schumann_field": np.full(8, 1.5, dtype=np.float64)})

    # An l6 payload with neither known key contributes a zero ecological effect.
    neutral = _layer(ecological_coupling=0.2)
    result = neutral.step(0.001, l6_input={"unrelated_channel": 1.0})
    assert "meridian_qi" in result


def test_l7_l6_drive_rejects_empty_payloads() -> None:
    layer = _layer()
    with pytest.raises(ValueError, match="schumann_field"):
        layer.step(0.001, l6_input={"schumann_field": np.array([], dtype=np.float64)})
    with pytest.raises(ValueError, match="symbolic_drive"):
        layer.step(0.001, l6_input={"symbolic_drive": np.array([], dtype=np.float64)})


def test_l7_acupoint_stimulus_rejects_non_integer_keys() -> None:
    layer = _layer()
    with pytest.raises(ValueError, match="integer point ids"):
        layer.step(0.001, acupoint_stimulus=cast(Any, {True: 0.5}))


def test_l7_symbol_input_rejects_non_finite_at_full_length() -> None:
    # A full-length symbol vector clears the size guard but a non-finite entry
    # is rejected by the finiteness check.
    layer = _layer()
    bad = np.ones(16, dtype=np.float64)
    bad[5] = np.nan
    with pytest.raises(ValueError, match="symbol_input must contain only finite"):
        layer.step(0.001, symbol_input=bad)


def test_l7_stimulate_meridian_guards() -> None:
    layer = _layer(n_meridians=4)
    with pytest.raises(ValueError, match="meridian_id must be in range"):
        layer.stimulate_meridian(10, 0.5)
    with pytest.raises(ValueError, match="intensity must be finite"):
        layer.stimulate_meridian(0, float("nan"))


def test_l7_validate_params_type_guards_and_negative_seed() -> None:
    with pytest.raises(ValueError, match="n_symbols must be a positive integer"):
        L7_SymbolicLayer(L7_StochasticParameters(n_symbols=cast(int, True)))
    with pytest.raises(ValueError, match="n_meridians must be a positive integer"):
        L7_SymbolicLayer(L7_StochasticParameters(n_meridians=cast(int, True)))
    with pytest.raises(ValueError, match="n_acupoints must be a positive integer"):
        L7_SymbolicLayer(L7_StochasticParameters(n_acupoints=cast(int, True)))
    with pytest.raises(ValueError, match="bitstream_length must be a positive integer"):
        L7_SymbolicLayer(L7_StochasticParameters(bitstream_length=cast(int, True)))
    with pytest.raises(ValueError, match="rng_seed"):
        L7_SymbolicLayer(L7_StochasticParameters(rng_seed=-1))
