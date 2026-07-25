# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing engine re-export contracts

"""Verify top-level quantum-annealing bindings and their downstream bridge flag."""

import pytest

from tests.sc_neurocore_engine_reexports_support import QA_SYMBOLS, _engine, _has_inner_qa


@pytest.mark.skipif(
    not _has_inner_qa(),
    reason="engine wheel built without QA bindings (rebuild via "
    "`cd bridge && maturin develop --release`)",
)
@pytest.mark.parametrize("sym", QA_SYMBOLS)
def test_qa_symbol_importable_from_toplevel(sym: str) -> None:
    """`from sc_neurocore_engine import py_qa_*` must resolve."""
    obj = getattr(_engine, sym, None)
    assert obj is not None, (
        f"{sym} not re-exported from sc_neurocore_engine.__init__; "
        f"check bridge/sc_neurocore_engine/__init__.py"
    )


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_qa_symbols_in_all() -> None:
    """All QA symbols must appear in the top-level `__all__` list."""
    assert hasattr(_engine, "__all__")
    public = set(_engine.__all__)
    missing = [symbol for symbol in QA_SYMBOLS if symbol not in public]
    assert not missing, f"QA symbols missing from __all__: {missing}"


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_qa_symbols_are_callable() -> None:
    """Every re-exported QA symbol must be callable rather than a stale stub."""
    for symbol in QA_SYMBOLS:
        obj = getattr(_engine, symbol)
        assert callable(obj), f"{symbol} is not callable: type={type(obj).__name__}"


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_qa_rust_available_flag_true() -> None:
    """The engine availability flag must mirror the compiled bindings."""
    assert getattr(_engine, "_qa_rust_available", False) is True


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_qa_simulated_annealing_returns_dict_keys() -> None:
    """Run the top-level binding and verify its stable result schema."""
    result = _engine.py_qa_simulated_annealing(
        [0, 1, 2, 3],
        [0.0, 0.0, 0.0, 0.0],
        [0, 1, 2],
        [1, 2, 3],
        [-1.0, -1.0, -1.0],
        4,
        0.0,
        100,
        2,
        0.1,
        5.0,
        42,
    )
    assert isinstance(result, dict)
    for key in ("best_spins", "best_energy"):
        assert key in result, f"missing key {key!r} in SA result; got {sorted(result)}"
    assert len(result["best_spins"]) == 4


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_bridges_quantum_annealing_HAS_RUST_QA_lit() -> None:
    """The consuming quantum-annealing bridge must detect the Rust backend."""
    from sc_neurocore.bridges.quantum_annealing import _HAS_RUST_QA

    assert _HAS_RUST_QA is True
