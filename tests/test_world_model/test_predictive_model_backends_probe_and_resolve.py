# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (probe_and_resolve) from former test_predictive_model_backends.py

from __future__ import annotations

from tests.test_world_model.predictive_model_backends_support import *  # noqa: F403


def test_probe_and_resolve_backend_follow_fastest_first_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    availability = {"mojo": False, "rust": False, "julia": True, "go": True}
    monkeypatch.setattr(backends, "_ensure_mojo_loaded", lambda: availability["mojo"])
    monkeypatch.setattr(backends, "_ensure_rust_loaded", lambda: availability["rust"])
    monkeypatch.setattr(backends, "_ensure_julia_loaded", lambda: availability["julia"])
    monkeypatch.setattr(backends, "_ensure_go_loaded", lambda: availability["go"])

    assert backends.probe_backend("python") == (True, "")
    assert backends.probe_backend("mojo")[0] is False
    assert "Mojo" in backends.probe_backend("mojo")[1]
    assert backends.resolve_backend("auto") == "go"
    assert backends.resolve_backend("go") == "go"

    availability["go"] = False
    with pytest.raises(RuntimeError, match="Go LGSSM backend"):
        backends.resolve_backend("go")
    with pytest.raises(ValueError, match="backend must be"):
        backends.resolve_backend("cuda")


def test_auto_resolution_reaches_python_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        backends,
        "probe_backend",
        lambda backend: (backend == "python", "missing"),
    )
    assert backends.resolve_backend("auto") == "python"


def test_auto_resolution_fails_closed_if_even_python_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backends,
        "probe_backend",
        lambda _backend: (False, "missing"),
    )
    with pytest.raises(RuntimeError, match="no executable LGSSM backend"):
        backends.resolve_backend("auto")
