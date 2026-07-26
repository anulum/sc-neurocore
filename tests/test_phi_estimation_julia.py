# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Phi Julia backend tests

"""Julia discovery and fail-closed Phi backend contracts."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403


def test_ensure_julia_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_julia_phi", None)
    monkeypatch.setattr(_PHI_MODULE._importlib_util, "find_spec", lambda _name: None)
    assert _PHI_MODULE._ensure_julia_phi() is False


def test_ensure_julia_false_when_module_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_julia_phi", None)
    monkeypatch.setattr(_PHI_MODULE._importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: False)
    assert _PHI_MODULE._ensure_julia_phi() is False


def test_julia_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_ensure_julia_phi", lambda: False)
    with pytest.raises(RuntimeError, match="Julia Phi backend is not available"):
        phi_star(_correlated(), tau=1, backend="julia")
