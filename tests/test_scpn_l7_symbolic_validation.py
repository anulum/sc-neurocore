# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L7 symbolic validation contracts

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l7_symbolic import L7_StochasticParameters, L7_SymbolicLayer
from tests.scpn_l7_symbolic_support import make_layer


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


def test_l7_acupoint_stimulus_rejects_non_integer_keys() -> None:
    layer = make_layer()
    with pytest.raises(ValueError, match="integer point ids"):
        layer.step(0.001, acupoint_stimulus=cast(Any, {True: 0.5}))


def test_l7_symbol_input_rejects_non_finite_at_full_length() -> None:
    # A full-length symbol vector clears the size guard but a non-finite entry
    # is rejected by the finiteness check.
    layer = make_layer()
    bad = np.ones(16, dtype=np.float64)
    bad[5] = np.nan
    with pytest.raises(ValueError, match="symbol_input must contain only finite"):
        layer.step(0.001, symbol_input=bad)


def test_l7_stimulate_meridian_guards() -> None:
    layer = make_layer(n_meridians=4)
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
