# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (julia_loader) from former test_predictive_model_backends.py

from __future__ import annotations

from tests.test_world_model.predictive_model_backends_support import *  # noqa: F403


def test_julia_loader_handles_cache_dependency_file_and_module_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(backends, "_julia_module", object())
    monkeypatch.setattr(backends, "_HAS_JULIA_LGSSM", False)
    assert backends._ensure_julia_loaded() is True
    assert backends._HAS_JULIA_LGSSM is True

    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(backends, "_HAS_JULIA_LGSSM", True)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert backends._ensure_julia_loaded() is False
    assert backends._HAS_JULIA_LGSSM is False

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(backends, "_PACKAGE_ROOT", tmp_path)
    assert backends._ensure_julia_loaded() is False

    module_path = tmp_path / "accel/julia/world_model/predictive_model.jl"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(Main=SimpleNamespace()),
    )
    assert backends._ensure_julia_loaded() is False

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(RuntimeError("Julia init failed")),
    )
    assert backends._ensure_julia_loaded() is False


def test_julia_loader_includes_module_and_caches_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_module = object()

    class Main:
        PredictiveModelAccel = loaded_module
        included: str | None = None

        @classmethod
        def include(cls, path: str) -> None:
            cls.included = path

    module_path = tmp_path / "accel/julia/world_model/predictive_model.jl"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    monkeypatch.setattr(backends, "_PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(Main=Main),
    )

    assert backends._ensure_julia_loaded() is True
    assert backends._julia_module is loaded_module
    assert Main.included == str(module_path)
    assert backends._HAS_JULIA_LGSSM is True
