# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L7 symbolic step contracts

from __future__ import annotations

import numpy as np

from tests.scpn_l7_symbolic_support import make_layer


def test_l7_step_consumes_valid_symbol_and_acupoint_inputs() -> None:
    layer = make_layer()
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
    layer = make_layer()
    layer.symbol_activations = np.zeros_like(layer.symbol_activations)
    layer.e8_state = np.zeros(8, dtype=np.float64)
    result = layer.step(0.001)
    # Sub-threshold activations and a zero E8 state take the neutral fallbacks.
    assert result["phi_alignment"] == 0.5
    assert result["fibonacci_alignment"] == 0.5
    assert result["e8_alignment"] == 0.5
