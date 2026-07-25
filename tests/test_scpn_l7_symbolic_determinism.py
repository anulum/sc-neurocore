# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L7 symbolic determinism and geometry contracts

from __future__ import annotations

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
