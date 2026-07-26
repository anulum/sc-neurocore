# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Phi Go backend tests

"""Go library discovery, symbol validation, and fail-closed Phi contracts."""

from __future__ import annotations

from tests.phi_estimation_support import *  # noqa: F403


def test_ensure_go_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_go_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: False)
    assert _PHI_MODULE._ensure_go_phi() is False


def test_ensure_go_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_go_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_PHI_MODULE._ctypes, "CDLL", _raise_oserror)
    assert _PHI_MODULE._ensure_go_phi() is False


def test_ensure_go_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_go_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_PHI_MODULE._ctypes, "CDLL", lambda _path: object())
    assert _PHI_MODULE._ensure_go_phi() is False


def test_go_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_PHI_MODULE, "_ensure_go_phi", lambda: False)
    with pytest.raises(RuntimeError, match="Go Phi backend is not available"):
        phi_star(_correlated(), tau=1, backend="go")
