# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rust_loader) from former test_predictive_model_backends.py

"""Rust filter loading for the predictive-model backends."""

from __future__ import annotations

import importlib
import importlib.util
import pytest
from sc_neurocore.world_model import (
    _lgssm_backends as backends,
)
from tests.test_world_model.predictive_model_backends_support import (
    _native_mapping,
)
from types import (
    SimpleNamespace,
)


def test_missing_rust_filter_raises() -> None:
    """The placeholder for a missing Rust filter raises."""
    with pytest.raises(RuntimeError, match="not available"):
        backends._missing_rust_kalman_filter()


def test_rust_loader_prefers_world_model_submodule(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Rust loader takes the filter from the world-model submodule first."""

    def sentinel(**_kwargs: object) -> object:
        return _native_mapping()

    def import_module(name: str) -> object:
        assert name == "sc_neurocore_engine.world_model"
        return SimpleNamespace(get_lgssm_kalman_filter=lambda: sentinel)

    monkeypatch.setattr(importlib, "import_module", import_module)
    assert backends._load_rust_kalman_filter() is sentinel


def test_rust_loader_uses_root_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the submodule the Rust loader falls back to the root export."""

    def sentinel(**_kwargs: object) -> object:
        return _native_mapping()

    def import_module(name: str) -> object:
        if name == "sc_neurocore_engine.world_model":
            raise ImportError(name)
        return SimpleNamespace(py_lgssm_kalman_filter=sentinel)

    monkeypatch.setattr(importlib, "import_module", import_module)
    assert backends._load_rust_kalman_filter() is sentinel


def test_rust_loader_rejects_missing_and_non_callable_exports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Rust loader refuses a missing or non-callable export."""
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )
    with pytest.raises(ImportError, match="not available"):
        backends._load_rust_kalman_filter()

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(get_lgssm_kalman_filter=lambda: object()),
    )
    with pytest.raises(ImportError, match="callable"):
        backends._load_rust_kalman_filter()


def test_module_initialisation_survives_missing_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backends module imports cleanly without the Rust engine."""
    real_import_module = importlib.import_module

    def import_module_without_rust(
        name: str,
        package: str | None = None,
    ) -> object:
        if name in {"sc_neurocore_engine", "sc_neurocore_engine.world_model"}:
            raise ImportError(name)
        return real_import_module(name, package)

    try:
        with monkeypatch.context() as context:
            context.setattr(importlib, "import_module", import_module_without_rust)
            reloaded = importlib.reload(backends)
            assert reloaded._rust_kalman_filter is None
            assert reloaded._HAS_RUST_LGSSM is False
    finally:
        importlib.reload(backends)


def test_ensure_rust_loaded_caches_success_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Rust probe caches a loaded filter and reports a failed load."""

    def sentinel(**_kwargs: object) -> object:
        return _native_mapping()

    def fail_to_load() -> object:
        raise ImportError("missing")

    monkeypatch.setattr(backends, "_rust_kalman_filter", sentinel)
    assert backends._ensure_rust_loaded() is True

    monkeypatch.setattr(backends, "_rust_kalman_filter", None)
    monkeypatch.setattr(
        backends,
        "_load_rust_kalman_filter",
        fail_to_load,
    )
    assert backends._ensure_rust_loaded() is False
    assert backends._HAS_RUST_LGSSM is False

    monkeypatch.setattr(backends, "_load_rust_kalman_filter", lambda: sentinel)
    assert backends._ensure_rust_loaded() is True
    assert backends._HAS_RUST_LGSSM is True
