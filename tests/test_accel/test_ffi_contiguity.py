# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Contiguity guards for native array boundaries

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.debug import sc_scope
from sc_neurocore.stochastic_doctor import diagnostics


class _UnexpectedRustCall:
    def py_scc_bytes(self, *_args, **_kwargs):
        raise AssertionError("Rust SCC path should not be reached for invalid layout")

    def py_precision_bytes(self, *_args, **_kwargs):
        raise AssertionError("Rust precision path should not be reached for invalid layout")

    def py_histogram(self, *_args, **_kwargs):
        raise AssertionError("Rust histogram path should not be reached for invalid layout")

    def py_scc_packed(self, *_args, **_kwargs):
        raise AssertionError("Rust packed SCC path should not be reached for invalid layout")


def test_diagnostics_compute_scc_rejects_non_contiguous_input(monkeypatch):
    monkeypatch.setattr(diagnostics, "_HAS_PYO3", True)
    monkeypatch.setattr(diagnostics, "_sdc_rust", _UnexpectedRustCall())

    bad = np.arange(16, dtype=np.uint8)[::2]
    good = np.ascontiguousarray(bad)

    with pytest.raises(ValueError, match="C-contiguous"):
        diagnostics.compute_scc(bad, good)


def test_diagnostics_precision_rejects_non_contiguous_input(monkeypatch):
    monkeypatch.setattr(diagnostics, "_HAS_PYO3", True)
    monkeypatch.setattr(diagnostics, "_sdc_rust", _UnexpectedRustCall())

    bad = np.arange(16, dtype=np.uint8)[::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        diagnostics.StochasticDoctor().estimate_precision(bad)


def test_diagnostics_histogram_rejects_non_contiguous_input(monkeypatch):
    monkeypatch.setattr(diagnostics, "_HAS_PYO3", True)
    monkeypatch.setattr(diagnostics, "_sdc_rust", _UnexpectedRustCall())

    bad = np.arange(64, dtype=np.uint8)[::2]

    with pytest.raises(ValueError, match="C-contiguous"):
        diagnostics.StochasticDoctor().compute_histogram(bad)


def test_sc_scope_rejects_non_contiguous_input_when_rust_enabled(monkeypatch):
    monkeypatch.setattr(sc_scope, "_HAS_RUST_SCC", True)
    monkeypatch.setattr(sc_scope, "_sdc", _UnexpectedRustCall())

    base = np.arange(32, dtype=np.uint32)
    bad = base[::2]
    good = np.ascontiguousarray(bad)

    with pytest.raises(ValueError, match="C-contiguous"):
        sc_scope.compute_scc(bad, good)
