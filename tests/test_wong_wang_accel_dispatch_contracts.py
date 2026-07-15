# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wong-Wang dispatcher failure contracts

"""Validate measured-order selection, inputs, results, and atomic failures."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.accel import wong_wang as dispatch
from sc_neurocore.neurons.models.wong_wang import WongWangUnit


def _valid_result(steps: int = 2) -> dict[str, object]:
    """Build one internally consistent backend mapping."""
    states = {
        "s1": np.linspace(0.1, 0.2, steps),
        "s2": np.linspace(0.1, 0.15, steps),
        "noise1": np.linspace(0.0, 0.01, steps),
        "noise2": np.linspace(0.0, -0.01, steps),
        "r1": np.linspace(2.0, 3.0, steps),
        "r2": np.linspace(1.0, 1.5, steps),
    }
    return {
        **states,
        "s1_final": 0.1 if steps == 0 else float(states["s1"][-1]),
        "s2_final": 0.1 if steps == 0 else float(states["s2"][-1]),
        "noise1_final": 0.0 if steps == 0 else float(states["noise1"][-1]),
        "noise2_final": 0.0 if steps == 0 else float(states["noise2"][-1]),
    }


def test_normaliser_accepts_complete_consistent_mapping() -> None:
    """Return contiguous typed traces only after all checks pass."""
    result = dispatch.normalise_result(_valid_result(), n_steps=2, initial=(0.1, 0.1, 0.0, 0.0))
    assert set(result) == {
        "s1",
        "s2",
        "noise1",
        "noise2",
        "r1",
        "r2",
        "s1_final",
        "s2_final",
        "noise1_final",
        "noise2_final",
    }
    for key in ("s1", "s2", "noise1", "noise2", "r1", "r2"):
        trace = result[key]
        assert isinstance(trace, np.ndarray)
        assert trace.flags.c_contiguous


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda result: result.pop("r2"), "invalid r2 trace"),
        (lambda result: result.__setitem__("s1", [[0.1], [0.2]]), "malformed s1 trace"),
        (lambda result: result.__setitem__("s2", [0.1]), "malformed s2 trace"),
        (lambda result: result.__setitem__("noise1", [0.0, np.nan]), "non-finite noise1"),
        (lambda result: result.__setitem__("s1", [0.1, 1.2]), "out-of-range s1"),
        (lambda result: result.__setitem__("r1", [2.0, -1.0]), "negative r1"),
        (lambda result: result.__setitem__("s2_final", object()), "invalid s2_final"),
        (lambda result: result.__setitem__("s1_final", 0.5), "s1_final disagrees"),
        (lambda result: result.__setitem__("noise2_final", np.inf), "non-finite noise2_final"),
    ),
)
def test_normaliser_rejects_malformed_results(
    mutation: Any,
    message: str,
) -> None:
    """Reject every externally observable shape and physics invariant."""
    result = _valid_result()
    mutation(result)
    with pytest.raises(FloatingPointError, match=message):
        dispatch.normalise_result(result, n_steps=2, initial=(0.1, 0.1, 0.0, 0.0))


@pytest.mark.parametrize(
    ("stim1", "stim2", "xi", "message"),
    (
        ([[0.0]], [0.0], [0.0, 0.0], "stim1 must be one-dimensional"),
        ([0.0], [[0.0]], [0.0, 0.0], "stim2 must be one-dimensional"),
        ([0.0], [0.0], [[0.0, 0.0]], "xi must be one-dimensional"),
        ([0.0, 0.0], [0.0], [0.0] * 4, "length mismatch"),
        ([0.0], [0.0], [0.0], "xi length"),
        ([np.nan], [0.0], [0.0, 0.0], "finite"),
    ),
)
def test_dispatcher_rejects_invalid_input_arrays(
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
    message: str,
) -> None:
    """Fail before any runtime is selected or called."""
    with pytest.raises(ValueError, match=message):
        dispatch.simulate_wong_wang(stim1=stim1, stim2=stim2, xi=xi, backend="python")


def test_unknown_backend_is_rejected() -> None:
    """Keep the backend name surface closed."""
    with pytest.raises(ValueError, match="unknown Wong-Wang backend"):
        dispatch.simulate_wong_wang(stim1=[], stim2=[], xi=[], backend="cuda")


def test_engine_and_native_module_loaders_resolve_the_named_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep dynamic import names and callable extraction explicit."""

    def runner(*_args: object) -> dict[str, object]:
        return _valid_result(0)

    engine_module = SimpleNamespace(py_wong_wang_simulate=runner)
    native_module = SimpleNamespace(simulate_wong_wang=runner)

    def import_module(name: str) -> object:
        if name == "sc_neurocore_engine":
            return engine_module
        assert name == "sc_neurocore.accel.go.wong_wang"
        return native_module

    monkeypatch.setattr(importlib, "import_module", import_module)
    assert dispatch._load_engine_runner() is runner
    assert dispatch._native_module("go") is native_module


def test_backend_availability_covers_every_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report only successfully loaded and explicitly enrolled runtimes."""

    def runner(*_args: object) -> dict[str, object]:
        return _valid_result(0)

    assert dispatch.backend_available("python")
    monkeypatch.setattr(dispatch, "_HAS_RUST", True)
    monkeypatch.setattr(dispatch, "_engine_simulate", runner)
    assert dispatch.backend_available("rust")
    monkeypatch.setattr(dispatch, "_engine_simulate", None)
    assert not dispatch.backend_available("rust")

    julia_module = SimpleNamespace(_ensure_wong_wang_loaded=lambda: None)
    monkeypatch.setattr(importlib, "import_module", lambda _name: julia_module)
    assert dispatch.backend_available("julia")

    def fail_julia() -> None:
        raise RuntimeError("Julia runtime is unavailable")

    julia_module._ensure_wong_wang_loaded = fail_julia
    assert not dispatch.backend_available("julia")

    compiled = SimpleNamespace(_HAS_GO_WONG_WANG=True, _HAS_MOJO_WONG_WANG=True)
    monkeypatch.setattr(dispatch, "_native_module", lambda _backend: compiled)
    assert dispatch.backend_available("go")
    assert dispatch.backend_available("mojo")
    compiled._HAS_GO_WONG_WANG = False
    assert not dispatch.backend_available("go")

    def missing_native(_backend: str) -> object:
        raise ImportError("native module is unavailable")

    monkeypatch.setattr(dispatch, "_native_module", missing_native)
    assert not dispatch.backend_available("mojo")
    assert not dispatch.backend_available("cuda")


def test_auto_backend_uses_measured_order_then_the_python_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select the first available measured lane and retain a safe floor."""
    observed: list[str] = []

    def availability(name: str) -> bool:
        observed.append(name)
        return name == "python"

    monkeypatch.setattr(
        dispatch, "select_backend_order", lambda *_args, **_kwargs: ("go", "python")
    )
    monkeypatch.setattr(dispatch, "backend_available", availability)
    assert dispatch.auto_backend() == "python"
    assert observed == ["go", "python"]

    monkeypatch.setattr(dispatch, "select_backend_order", lambda *_args, **_kwargs: ("go",))
    monkeypatch.setattr(dispatch, "backend_available", lambda _name: False)
    assert dispatch.auto_backend() == "python"


def test_native_runner_resolves_each_boundary_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve Rust, Julia, and C-backed facades without implicit fallback."""

    def runner(*_args: object) -> dict[str, object]:
        return _valid_result(0)

    monkeypatch.setattr(dispatch, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="Rust Wong-Wang backend is unavailable"):
        dispatch._native_runner("rust")
    monkeypatch.setattr(dispatch, "_engine_simulate", runner)
    assert dispatch._native_runner("rust") is runner

    julia_module = SimpleNamespace(simulate_wong_wang=runner)
    monkeypatch.setattr(importlib, "import_module", lambda _name: julia_module)
    assert dispatch._native_runner("julia") is runner

    native_module = SimpleNamespace(simulate_wong_wang=runner)
    monkeypatch.setattr(dispatch, "_native_module", lambda _backend: native_module)
    assert dispatch._native_runner("go") is runner


@pytest.mark.parametrize("backend", ("rust", "julia", "go", "mojo"))
def test_explicit_unavailable_backend_fails_closed(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never silently fall back after an explicit runtime request."""
    monkeypatch.setattr(dispatch, "backend_available", lambda name: name != backend)
    with pytest.raises(RuntimeError, match=f"{backend} Wong-Wang backend is unavailable"):
        dispatch.simulate_wong_wang(stim1=[], stim2=[], xi=[], backend=backend)


def test_auto_uses_the_selected_available_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise selection separately from numerical parity."""
    monkeypatch.setattr(dispatch, "auto_backend", lambda: "python")
    result = dispatch.simulate_wong_wang(stim1=[], stim2=[], xi=[], backend="auto")
    assert result["s1_final"] == 0.1
    assert result["noise2_final"] == 0.0


def test_malformed_native_result_does_not_mutate_public_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit model state only after shared result validation."""
    unit = WongWangUnit(noise1=0.01, noise2=-0.02)
    before = (unit.s1, unit.s2, unit.noise1, unit.noise2)

    def runner(*_args: object) -> dict[str, object]:
        result = _valid_result(1)
        result["s1_final"] = 0.5
        return result

    monkeypatch.setattr(dispatch, "backend_available", lambda _name: True)
    monkeypatch.setattr(dispatch, "_native_runner", lambda _name: runner)
    with pytest.raises(FloatingPointError, match="s1_final disagrees"):
        unit.simulate([0.0], [0.0], [0.0, 0.0], backend="go")
    assert (unit.s1, unit.s2, unit.noise1, unit.noise2) == before


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_ctypes_facades_share_shape_and_finite_input_contracts(backend: str) -> None:
    """Keep both C-backed Python boundaries aligned with the dispatcher."""
    module = importlib.import_module(f"sc_neurocore.accel.{backend}.wong_wang")
    with pytest.raises(ValueError, match="stim1 must be one-dimensional"):
        module.simulate_wong_wang(
            0.1,
            0.1,
            0.0,
            0.0,
            0.1,
            0.002,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.02,
            0.0001,
            [[0.0]],
            [0.0],
            [0.0, 0.0],
        )
    with pytest.raises(ValueError, match="xi length"):
        module.simulate_wong_wang(
            0.1,
            0.1,
            0.0,
            0.0,
            0.1,
            0.002,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.02,
            0.0001,
            [0.0],
            [0.0],
            [0.0],
        )


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_ctypes_loader_failure_sets_unavailable_sentinel(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose missing shared objects as availability state, not import failure."""
    module = importlib.import_module(f"sc_neurocore.accel.{backend}.wong_wang")
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _path: (_ for _ in ()).throw(OSError()))
    assert module._load_library() == (None, False)
