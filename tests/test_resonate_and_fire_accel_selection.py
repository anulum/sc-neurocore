# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Resonate-and-fire accelerator selection contracts

"""Cover input validation, backend discovery, selection, and module reload."""

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sc_neurocore.accel import resonate_and_fire as backends
from tests.module_reload import preserve_module_identity
from tests.resonate_and_fire_accel_dispatch_support import _PARAMETERS


def test_input_shape_and_finiteness_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        backends.simulate_resonate_and_fire(*_PARAMETERS, np.zeros((2, 2)), backend="python")
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_resonate_and_fire(*_PARAMETERS, [1.5, np.nan], backend="python")


def test_signed_32_bit_step_bound_precedes_contiguous_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = np.broadcast_to(np.asarray([1.0]), ((1 << 31),))

    def unexpected_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized logical input reached contiguous allocation")

    monkeypatch.setattr(np, "ascontiguousarray", unexpected_copy)
    with pytest.raises(ValueError, match="signed-32-bit step limit"):
        backends._input(oversized)


def test_unknown_and_explicitly_unavailable_backends_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unknown resonate-and-fire backend"):
        backends.simulate_resonate_and_fire(*_PARAMETERS, [1.5], backend="fortran")
    monkeypatch.setattr(backends, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="Rust resonate-and-fire backend is unavailable"):
        backends.simulate_resonate_and_fire(*_PARAMETERS, [1.5], backend="rust")


def test_auto_selection_uses_first_available_measured_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backends,
        "select_backend_order",
        lambda _kernel, static: ("mojo", "go", "rust", "julia", static[-1]),
    )
    monkeypatch.setattr(
        backends,
        "backend_available",
        lambda backend: backend in {"go", "python"},
    )
    assert backends.auto_backend() == "go"


def test_python_floor_is_always_available() -> None:
    assert backends.backend_available("python")
    assert not backends.backend_available("unknown")


def test_optional_backend_discovery_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    class JuliaError(Exception):
        """Stand in for JuliaCall's optional runtime exception type."""

    monkeypatch.setattr(
        backends,
        "_ensure_julia_loaded",
        lambda: (_ for _ in ()).throw(JuliaError("Julia startup failed")),
    )
    assert not backends.backend_available("julia")

    monkeypatch.setattr(
        backends,
        "_ensure_julia_loaded",
        lambda: (_ for _ in ()).throw(RuntimeError("unrelated Python defect")),
    )
    with pytest.raises(RuntimeError, match="unrelated Python defect"):
        backends.backend_available("julia")

    def missing_native(_backend: str) -> Any:
        raise ImportError("native module absent")

    monkeypatch.setattr(backends, "_native_module", missing_native)
    assert not backends.backend_available("go")


def test_missing_engine_export_disables_rust_without_breaking_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_simulate_python = backends.simulate_python
    real_import_module = importlib.import_module

    def import_without_export(name: str) -> Any:
        if name == "sc_neurocore_engine":
            return SimpleNamespace()
        return real_import_module(name)

    with preserve_module_identity(backends), monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", import_without_export)
        reloaded = importlib.reload(backends)
        assert not reloaded.backend_available("rust")
        assert reloaded.backend_available("python")

    assert backends.simulate_python is original_simulate_python
    assert np.asarray(backends.simulate_python(*_PARAMETERS, [1.5])["x"]).shape == (1,)
