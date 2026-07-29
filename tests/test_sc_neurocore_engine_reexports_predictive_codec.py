# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive-codec engine re-export contracts

"""Verify the top-level predictive-codec bindings and availability flag."""

import pytest

from tests.sc_neurocore_engine_reexports_support import (
    PREDICTIVE_CODEC_SYMBOLS,
    _engine,
)


@pytest.mark.parametrize("sym", PREDICTIVE_CODEC_SYMBOLS)
def test_predictive_codec_symbol_importable_from_toplevel(sym: str) -> None:
    obj = getattr(_engine, sym, None)
    assert obj is not None, f"{sym} not re-exported from sc_neurocore_engine.__init__"


def test_predictive_codec_symbols_in_all() -> None:
    public = set(_engine.__all__)
    missing = [symbol for symbol in PREDICTIVE_CODEC_SYMBOLS if symbol not in public]
    assert not missing, f"Predictive codec symbols missing from __all__: {missing}"


def test_predictive_codec_symbols_are_callable() -> None:
    for symbol in PREDICTIVE_CODEC_SYMBOLS:
        obj = getattr(_engine, symbol)
        assert callable(obj), f"{symbol} is not callable: type={type(obj).__name__}"


def test_predictive_codec_rust_available_flag_true() -> None:
    assert getattr(_engine, "_predictive_codec_rust_available", False) is True
