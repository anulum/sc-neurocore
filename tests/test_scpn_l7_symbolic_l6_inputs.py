# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L7 symbolic L6 input contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l7_symbolic import L7_StochasticParameters, L7_SymbolicLayer
from tests.scpn_l7_symbolic_support import make_layer


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


def test_l7_l6_schumann_fallback_and_neutral_payload() -> None:
    # schumann_field (no symbolic_drive) drives the finite-mean fallback branch.
    schumann = make_layer(ecological_coupling=0.2)
    schumann.step(0.001, l6_input={"schumann_field": np.full(8, 1.5, dtype=np.float64)})

    # An l6 payload with neither known key contributes a zero ecological effect.
    neutral = make_layer(ecological_coupling=0.2)
    result = neutral.step(0.001, l6_input={"unrelated_channel": 1.0})
    assert "meridian_qi" in result


def test_l7_l6_drive_rejects_empty_payloads() -> None:
    layer = make_layer()
    with pytest.raises(ValueError, match="schumann_field"):
        layer.step(0.001, l6_input={"schumann_field": np.array([], dtype=np.float64)})
    with pytest.raises(ValueError, match="symbolic_drive"):
        layer.step(0.001, l6_input={"symbolic_drive": np.array([], dtype=np.float64)})
