# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA-mapper engine re-export contracts

"""Verify top-level DNA bindings and their downstream bridge flag."""

import pytest

from tests.sc_neurocore_engine_reexports_support import DNA_SYMBOLS, _engine, _has_inner_dna


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
@pytest.mark.parametrize("sym", DNA_SYMBOLS)
def test_dna_symbol_importable_from_toplevel(sym: str) -> None:
    """`from sc_neurocore_engine import py_dna_*` must resolve."""
    obj = getattr(_engine, sym, None)
    assert obj is not None, f"{sym} not re-exported from sc_neurocore_engine.__init__"


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
def test_dna_symbols_in_all() -> None:
    public = set(_engine.__all__)
    missing = [symbol for symbol in DNA_SYMBOLS if symbol not in public]
    assert not missing, f"DNA symbols missing from __all__: {missing}"


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
def test_dna_symbols_are_callable() -> None:
    for symbol in DNA_SYMBOLS:
        obj = getattr(_engine, symbol)
        assert callable(obj), f"{symbol} is not callable: type={type(obj).__name__}"


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
def test_dna_rust_available_flag_true() -> None:
    assert getattr(_engine, "_dna_rust_available", False) is True


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
def test_bridges_dna_mapper_HAS_RUST_DNA_lit() -> None:
    """The consuming DNA mapper bridge must detect the Rust backend."""
    from sc_neurocore.bridges.dna_mapper import _HAS_RUST_DNA

    assert _HAS_RUST_DNA is True
