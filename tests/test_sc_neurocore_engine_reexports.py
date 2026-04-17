# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore_engine top-level re-exports

"""Verify the bridge `__init__.py` re-exports for the engine wheel.

Commit `394f38da` added `try: from sc_neurocore_engine import py_qa_*`
re-exports for 6 quantum-annealing PyO3 bindings + 5 DNA-mapper
bindings, so that the bridges' `_HAS_RUST_QA` / `_HAS_RUST_DNA`
flags resolve to True when the engine wheel is installed.

This file is the test that proves the wiring works:
- All 11 symbols are importable from the top-level
  `sc_neurocore_engine` namespace (not just from the inner
  `sc_neurocore_engine.sc_neurocore_engine` module).
- They are present in `__all__`.
- They are callable (we don't validate semantics here — that
  belongs in `tests/test_bridges/test_quantum_annealing*.py`
  and equivalent for DNA — but a callable smoke proves the
  binding is not a stale stub).

Skips cleanly when the engine wheel is not present (e.g. CI
without `maturin develop` in the active venv).
"""

from __future__ import annotations

import importlib

import pytest


_engine = pytest.importorskip("sc_neurocore_engine")

QA_SYMBOLS: tuple[str, ...] = (
    "py_qa_batch_ising_energy",
    "py_qa_gauge_transform",
    "py_qa_generate_gauges",
    "py_qa_greedy_partition",
    "py_qa_ising_energy",
    "py_qa_simulated_annealing",
)

DNA_SYMBOLS: tuple[str, ...] = (
    "py_dna_check_cross_hybridization",
    "py_dna_design_orthogonal_set",
    "py_dna_design_sequence",
    "py_dna_detect_hairpins",
    "py_dna_simulate_kinetics",
)


def _has_inner_qa() -> bool:
    """True when the inner Rust module exposes the QA bindings."""
    try:
        inner = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return False
    return all(hasattr(inner, sym) for sym in QA_SYMBOLS)


def _has_inner_dna() -> bool:
    try:
        inner = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    except ImportError:
        return False
    return all(hasattr(inner, sym) for sym in DNA_SYMBOLS)


# ───────────────────────── QA re-exports ─────────────────────────


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
    missing = [s for s in QA_SYMBOLS if s not in public]
    assert not missing, f"QA symbols missing from __all__: {missing}"


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_qa_symbols_are_callable() -> None:
    """Every re-exported QA symbol must be callable (not a stale stub)."""
    for sym in QA_SYMBOLS:
        obj = getattr(_engine, sym)
        assert callable(obj), f"{sym} is not callable: type={type(obj).__name__}"


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_qa_rust_available_flag_true() -> None:
    """`_qa_rust_available` flag in the engine init must mirror the wiring."""
    assert getattr(_engine, "_qa_rust_available", False) is True


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_qa_simulated_annealing_returns_dict_keys() -> None:
    """End-to-end smoke: run py_qa_simulated_annealing, check schema.

    Uses a 4-qubit toy Ising (h=0, single FM bond). We don't assert
    on exact ground state — the SA bug-fix tests live elsewhere —
    only on the result schema.
    """
    h_indices = [0, 1, 2, 3]
    h_values = [0.0, 0.0, 0.0, 0.0]
    j_i = [0, 1, 2]
    j_j = [1, 2, 3]
    j_values = [-1.0, -1.0, -1.0]

    result = _engine.py_qa_simulated_annealing(
        h_indices,
        h_values,
        j_i,
        j_j,
        j_values,
        4,    # n_qubits
        0.0,  # offset
        100,  # n_sweeps
        2,    # num_reads
        0.1,  # beta_start
        5.0,  # beta_end
        42,   # seed
    )
    assert isinstance(result, dict)
    for key in ("best_spins", "best_energy"):
        assert key in result, f"missing key {key!r} in SA result; got {sorted(result)}"
    assert len(result["best_spins"]) == 4


# ───────────────────────── DNA re-exports ─────────────────────────


@pytest.mark.skipif(
    not _has_inner_dna(),
    reason="engine wheel built without DNA bindings",
)
@pytest.mark.parametrize("sym", DNA_SYMBOLS)
def test_dna_symbol_importable_from_toplevel(sym: str) -> None:
    """`from sc_neurocore_engine import py_dna_*` must resolve."""
    obj = getattr(_engine, sym, None)
    assert obj is not None, (
        f"{sym} not re-exported from sc_neurocore_engine.__init__"
    )


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
def test_dna_symbols_in_all() -> None:
    public = set(_engine.__all__)
    missing = [s for s in DNA_SYMBOLS if s not in public]
    assert not missing, f"DNA symbols missing from __all__: {missing}"


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
def test_dna_symbols_are_callable() -> None:
    for sym in DNA_SYMBOLS:
        obj = getattr(_engine, sym)
        assert callable(obj), f"{sym} is not callable: type={type(obj).__name__}"


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
def test_dna_rust_available_flag_true() -> None:
    assert getattr(_engine, "_dna_rust_available", False) is True


# ───────────────────────── Bridge flag wiring ─────────────────────────


@pytest.mark.skipif(not _has_inner_qa(), reason="engine wheel built without QA bindings")
def test_bridges_quantum_annealing_HAS_RUST_QA_lit() -> None:
    """Downstream consumer: bridges.quantum_annealing._HAS_RUST_QA must flip True."""
    from sc_neurocore.bridges.quantum_annealing import _HAS_RUST_QA
    assert _HAS_RUST_QA is True


@pytest.mark.skipif(not _has_inner_dna(), reason="engine wheel built without DNA bindings")
def test_bridges_dna_mapper_HAS_RUST_DNA_lit() -> None:
    """Downstream consumer: bridges.dna_mapper._HAS_RUST_DNA must flip True."""
    from sc_neurocore.bridges.dna_mapper import _HAS_RUST_DNA
    assert _HAS_RUST_DNA is True
