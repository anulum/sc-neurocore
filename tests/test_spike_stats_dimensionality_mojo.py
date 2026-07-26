# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Spike-statistics Mojo dimensionality tests

"""Mojo library discovery, symbol validation, and fail-closed backend contracts."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403


def test_ensure_mojo_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_mojo_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: False)
    assert _DIM._ensure_mojo_dim() is False


def test_ensure_mojo_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_mojo_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_DIM._ctypes, "CDLL", _raise_oserror)
    assert _DIM._ensure_mojo_dim() is False


def test_ensure_mojo_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_mojo_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_DIM._ctypes, "CDLL", lambda _path: object())
    assert _DIM._ensure_mojo_dim() is False


def test_mojo_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_ensure_mojo_dim", lambda: False)
    with pytest.raises(RuntimeError, match="Mojo dimensionality backend is not available"):
        spike_train_pca(_trains(), backend="mojo")
    with pytest.raises(RuntimeError, match="Mojo dimensionality backend is not available"):
        demixed_pca(_conditions(), backend="mojo")
    with pytest.raises(RuntimeError, match="Mojo dimensionality backend is not available"):
        factor_analysis(_trains(5), backend="mojo")
