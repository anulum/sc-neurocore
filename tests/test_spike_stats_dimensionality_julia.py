# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Spike-statistics Julia dimensionality tests

"""Julia discovery and fail-closed dimensionality backend contracts."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403


def test_ensure_julia_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_julia_dim", None)
    monkeypatch.setattr(_DIM._importlib_util, "find_spec", lambda _name: None)
    assert _DIM._ensure_julia_dim() is False


def test_ensure_julia_false_when_module_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_julia_dim", None)
    monkeypatch.setattr(_DIM._importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: False)
    assert _DIM._ensure_julia_dim() is False


def test_julia_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_ensure_julia_dim", lambda: False)
    with pytest.raises(RuntimeError, match="Julia dimensionality backend is not available"):
        spike_train_pca(_trains(), backend="julia")
    with pytest.raises(RuntimeError, match="Julia dimensionality backend is not available"):
        demixed_pca(_conditions(), backend="julia")
    with pytest.raises(RuntimeError, match="Julia dimensionality backend is not available"):
        factor_analysis(_trains(5), backend="julia")
