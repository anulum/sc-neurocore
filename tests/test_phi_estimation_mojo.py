# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Phi Mojo backend tests

"""Mojo library discovery, symbol validation, and fail-closed Phi contracts."""

from __future__ import annotations

import pytest
from sc_neurocore.analysis.phi_estimation import (
    phi_star,
)
from tests.phi_estimation_support import (
    _correlated,
    _PHI_MODULE,
    _raise_oserror,
)


def test_ensure_mojo_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without its library the Mojo lane is off."""
    monkeypatch.setattr(_PHI_MODULE, "_mojo_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: False)
    assert _PHI_MODULE._ensure_mojo_phi() is False


def test_ensure_mojo_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Mojo load error leaves the lane off."""
    monkeypatch.setattr(_PHI_MODULE, "_mojo_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_PHI_MODULE._ctypes, "CDLL", _raise_oserror)
    assert _PHI_MODULE._ensure_mojo_phi() is False


def test_ensure_mojo_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Mojo symbol leaves the lane off."""
    monkeypatch.setattr(_PHI_MODULE, "_mojo_phi_lib", None)
    monkeypatch.setattr(_PHI_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_PHI_MODULE._ctypes, "CDLL", lambda _path: object())
    assert _PHI_MODULE._ensure_mojo_phi() is False


def test_mojo_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requesting the missing Mojo lane raises."""
    monkeypatch.setattr(_PHI_MODULE, "_ensure_mojo_phi", lambda: False)
    with pytest.raises(RuntimeError, match="Mojo Phi backend is not available"):
        phi_star(_correlated(), tau=1, backend="mojo")
