# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Branch tests for stochastic doctor diagnostics

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore.stochastic_doctor.diagnostics as diag


@pytest.fixture(autouse=True)
def _restore_diag_state() -> None:
    old_has = diag._HAS_PYO3
    old_rust = diag._sdc_rust
    try:
        yield
    finally:
        diag._HAS_PYO3 = old_has
        diag._sdc_rust = old_rust


def test_scc_python_zero_numerator_branch() -> None:
    a = np.array([1, 0, 1, 0], dtype=np.uint8)
    b = np.array([1, 0, 1, 0], dtype=np.uint8)
    assert diag._scc_python(a, b) == pytest.approx(1.0)

    c = np.array([1, 0, 1, 0], dtype=np.uint8)
    d = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert diag._scc_python(c, d) == pytest.approx(0.0)


def test_compute_scc_uses_rust_when_available() -> None:
    class _FakeRust:
        @staticmethod
        def py_scc_bytes(a: np.ndarray, b: np.ndarray) -> float:
            assert a.flags.c_contiguous
            assert b.flags.c_contiguous
            return 0.125

    diag._HAS_PYO3 = True
    diag._sdc_rust = _FakeRust()

    a = np.asfortranarray(np.array([1, 0, 1, 0], dtype=np.uint8))
    b = np.asfortranarray(np.array([0, 1, 0, 1], dtype=np.uint8))
    assert diag.compute_scc(a, b) == pytest.approx(0.125)


def test_estimate_precision_handles_empty_and_rust_path() -> None:
    doctor = diag.StochasticDoctor()
    assert doctor.estimate_precision(np.array([], dtype=np.uint8)) == (0.0, 0.0)

    class _FakeRust:
        @staticmethod
        def py_precision_bytes(bitstream: np.ndarray) -> tuple[float, float]:
            assert bitstream.flags.c_contiguous
            return (0.3, 0.01)

    diag._HAS_PYO3 = True
    diag._sdc_rust = _FakeRust()
    bs = np.asfortranarray(np.array([1, 0, 1], dtype=np.uint8))
    assert doctor.estimate_precision(bs) == (0.3, 0.01)


def test_compute_histogram_rust_path() -> None:
    class _FakeRust:
        @staticmethod
        def py_histogram(bitstream: np.ndarray, word_size: int) -> list[int]:
            assert bitstream.flags.c_contiguous
            return [0] * word_size + [2]

    diag._HAS_PYO3 = True
    diag._sdc_rust = _FakeRust()

    doctor = diag.StochasticDoctor()
    hist = doctor.compute_histogram(np.asfortranarray(np.ones(8, dtype=np.uint8)), word_size=8)
    assert hist[-1] == 2


def test_audit_layer_warning_status(monkeypatch: pytest.MonkeyPatch) -> None:
    doctor = diag.StochasticDoctor(correlation_threshold=0.2, critical_threshold=0.95)
    monkeypatch.setattr(diag, "compute_scc", lambda _a, _b: 0.5)
    a = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
    b = np.array([1, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    report = doctor.audit_layer("warn", np.stack([a, b]))
    assert report.status is diag.AuditSeverity.WARNING
    assert report.findings
