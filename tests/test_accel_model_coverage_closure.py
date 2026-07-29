# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused accelerator model and language-facade coverage

"""Close remaining accelerator model, loader, and language-facade branches."""

from __future__ import annotations

import importlib
import inspect
import math
import sys
from types import ModuleType

import numpy as np
import pytest

from sc_neurocore.accel import (
    adaptive_threshold_if,
    coba_lif,
    ermentrout_kopell_pop,
    gpu_backend,
    iqif,
    mcculloch_pitts,
    resonate_and_fire,
)
from sc_neurocore.accel.go import alpha as go_alpha
from sc_neurocore.accel.go import wilson_cowan as go_wilson_cowan
from sc_neurocore.accel.go import wong_wang as go_wong_wang
from sc_neurocore.accel.julia.neurons import adaptive_threshold_if as julia_adaptive
from sc_neurocore.accel.julia.neurons import alpha as julia_alpha
from sc_neurocore.accel.mojo import alpha as mojo_alpha
from sc_neurocore.accel.mojo import wilson_cowan as mojo_wilson_cowan
from sc_neurocore.accel.mojo import wong_wang as mojo_wong_wang


@pytest.mark.parametrize(
    ("module_name", "engine_field"),
    [
        ("sc_neurocore.accel.coba_lif", "_engine_coba_simulate"),
        ("sc_neurocore.accel.jansen_rit", "_engine_simulate"),
        ("sc_neurocore.accel.wong_wang", "_engine_simulate"),
    ],
)
def test_model_accelerator_import_falls_back_without_rust_engine(
    monkeypatch: pytest.MonkeyPatch, module_name: str, engine_field: str
) -> None:
    """Import-time Rust absence leaves a deterministic unavailable marker."""

    original_import = importlib.import_module
    module = original_import(module_name)

    def import_without_engine(name: str, package: str | None = None) -> ModuleType:
        if name == "sc_neurocore_engine":
            raise ImportError("engine unavailable")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_engine)
    module = importlib.reload(sys.modules[module.__name__])

    assert module._HAS_RUST is False
    assert getattr(module, engine_field) is None
    monkeypatch.undo()
    importlib.reload(module)


def test_coba_loaders_fail_closed_at_every_external_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COBA Julia, Go, and Mojo discovery never exposes partial runtimes."""

    monkeypatch.setattr(coba_lif, "_julia_module", None)
    monkeypatch.setattr(coba_lif.importlib_util, "find_spec", lambda _name: None)
    assert coba_lif.ensure_julia_loaded() is False
    monkeypatch.setattr(coba_lif.importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(coba_lif.os.path, "isfile", lambda _path: False)
    assert coba_lif.ensure_julia_loaded() is False
    monkeypatch.setattr(coba_lif.os.path, "isfile", lambda _path: True)

    def fail_import(_name: str) -> ModuleType:
        raise RuntimeError("Julia load failed")

    monkeypatch.setattr(coba_lif.importlib, "import_module", fail_import)
    assert coba_lif.ensure_julia_loaded() is False
    assert coba_lif._configure_c_library(object(), "missing", mojo=False) is None

    for state_name, loader_name in (
        ("_go_lib", "ensure_go_loaded"),
        ("_mojo_lib", "ensure_mojo_loaded"),
    ):
        monkeypatch.setattr(coba_lif, state_name, None)
        monkeypatch.setattr(coba_lif.os.path, "isfile", lambda _path: False)
        assert getattr(coba_lif, loader_name)() is False
        monkeypatch.setattr(coba_lif.os.path, "isfile", lambda _path: True)

        def fail_cdll(_path: str) -> object:
            raise OSError("shared object rejected")

        monkeypatch.setattr(coba_lif.ctypes, "CDLL", fail_cdll)
        assert getattr(coba_lif, loader_name)() is False


@pytest.mark.parametrize(
    ("module", "function_name", "state_name"),
    [
        *[
            (coba_lif, function_name, state_name)
            for function_name, state_name in (
                ("simulate_rust", "_engine_coba_simulate"),
                ("simulate_julia", "_julia_module"),
                ("simulate_go", "_go_lib"),
                ("simulate_mojo", "_mojo_lib"),
            )
        ],
        *[
            (iqif, function_name, state_name)
            for function_name, state_name in (
                ("simulate_rust", "_engine_simulate"),
                ("simulate_julia", "_julia_module"),
                ("simulate_go", "_go_lib"),
                ("simulate_mojo", "_mojo_lib"),
            )
        ],
        *[
            (mcculloch_pitts, function_name, state_name)
            for function_name, state_name in (
                ("evaluate_rust", "_engine_evaluate"),
                ("evaluate_julia", "_julia_module"),
                ("evaluate_go", "_go_lib"),
                ("evaluate_mojo", "_mojo_lib"),
            )
        ],
    ],
)
def test_remaining_native_facades_reject_unavailable_backends(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    function_name: str,
    state_name: str,
) -> None:
    """Direct model facades check availability before touching arguments."""

    monkeypatch.setattr(module, state_name, None)
    function = getattr(module, function_name)
    arguments = [0 for _ in inspect.signature(function).parameters]

    with pytest.raises(RuntimeError, match="unavailable"):
        function(*arguments)


def test_model_backend_inventory_handles_absent_julia_and_unknown_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model availability probes convert known runtime absence to false."""

    monkeypatch.setattr(
        adaptive_threshold_if,
        "_ensure_julia_loaded",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    monkeypatch.setattr(
        resonate_and_fire,
        "_ensure_julia_loaded",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    assert adaptive_threshold_if.backend_available("julia") is False
    assert resonate_and_fire.backend_available("julia") is False

    def fail_import(_name: str) -> ModuleType:
        raise ImportError("missing")

    monkeypatch.setattr(ermentrout_kopell_pop.importlib, "import_module", fail_import)
    assert ermentrout_kopell_pop.backend_available("julia") is False
    assert iqif.backend_available("unknown") is False
    assert mcculloch_pitts.backend_available("unknown") is False


def test_adaptive_and_resonate_receipts_reject_nonfinite_and_missing_finals() -> None:
    """Trace receipts require finite final state and an explicit spike count."""

    adaptive = {
        "v": [],
        "theta": [],
        "spikes": [],
        "v_final": math.nan,
        "theta_final": 0.0,
        "spike_count": 0,
    }
    with pytest.raises(FloatingPointError, match="non-finite v_final"):
        adaptive_threshold_if.normalise_result(
            adaptive,
            n_steps=0,
            initial=(0.0, 0.0),
            v_reset=0.0,
            theta_rest=0.0,
            delta_theta=0.0,
            tau_theta=1.0,
            dt=0.1,
        )
    adaptive["v_final"] = 0.0
    adaptive.pop("spike_count")
    with pytest.raises(FloatingPointError, match="invalid spike_count"):
        adaptive_threshold_if.normalise_result(
            adaptive,
            n_steps=0,
            initial=(0.0, 0.0),
            v_reset=0.0,
            theta_rest=0.0,
            delta_theta=0.0,
            tau_theta=1.0,
            dt=0.1,
        )

    resonate = {
        "x": [],
        "y": [],
        "spikes": [],
        "x_final": math.nan,
        "y_final": 0.0,
        "spike_count": 0,
    }
    with pytest.raises(FloatingPointError, match="non-finite x_final"):
        resonate_and_fire.normalise_result(resonate, n_steps=0, initial=(0.0, 0.0), threshold=1.0)
    resonate["x_final"] = 0.0
    resonate.pop("spike_count")
    with pytest.raises(FloatingPointError, match="invalid spike_count"):
        resonate_and_fire.normalise_result(resonate, n_steps=0, initial=(0.0, 0.0), threshold=1.0)


@pytest.mark.parametrize(
    ("trace", "spikes", "final_v", "match"),
    [
        ([object()], 0, 0, "non-numeric"),
        ([0], 0, True, "invalid final voltage"),
        ([0], object(), 0, "invalid spike count"),
        ([0], 2, 0, "invalid spike count"),
    ],
)
def test_iqif_result_rejects_malformed_native_receipts(
    trace: object, spikes: object, final_v: object, match: str
) -> None:
    """IQIF requires integral bounded state, count, and final receipts."""

    with pytest.raises(FloatingPointError, match=match):
        iqif.normalise_result(trace, spikes, final_v, n_steps=1, v_min=-10, v_max=10)


def test_mcculloch_pitts_rejects_non_numeric_output() -> None:
    """Binary event results cannot contain non-numeric trace or count objects."""

    with pytest.raises(FloatingPointError, match="non-numeric"):
        mcculloch_pitts.normalise_result(([object()], object()), expected_length=1)


@pytest.mark.parametrize("module", (go_alpha, mojo_alpha))
def test_alpha_native_facades_reject_mismatched_inhibitory_drive(module: ModuleType) -> None:
    """Alpha facades require scalar or exact-length inhibitory current."""

    with pytest.raises(ValueError, match="scalar or match"):
        module.simulate_alpha(*([0.0] * 11), [], [1.0])


@pytest.mark.parametrize("module", (go_wong_wang, mojo_wong_wang))
def test_wong_wang_native_facades_reject_wrong_noise_length(module: ModuleType) -> None:
    """Wong-Wang facades require two noise samples per model step."""

    with pytest.raises(ValueError, match="xi length"):
        module._inputs([], [], [1.0])
    with pytest.raises(ValueError, match="stim1 and stim2 length mismatch"):
        module._inputs([], [0.0], [])


@pytest.mark.parametrize("module", (go_wilson_cowan, mojo_wilson_cowan))
def test_wilson_cowan_native_facades_reject_invalid_public_contracts(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Wilson-Cowan facades reject drive, weight, time, and state violations."""

    with pytest.raises(ValueError, match="finite values"):
        module._as_ext_input([math.nan])
    monkeypatch.setattr(module, "_MAX_STEPS", 0)
    with pytest.raises(ValueError, match="at most"):
        module._as_ext_input([0.0])
    with pytest.raises(ValueError, match="configuration must be finite"):
        module._validate_configuration(math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        module._validate_configuration(0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="must be positive"):
        module._validate_configuration(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="state envelope"):
        module._validate_configuration(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.1)


@pytest.mark.parametrize("module", (julia_adaptive, julia_alpha))
def test_julia_model_facades_reject_missing_runtime_source_and_bad_input(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Julia facades reject missing runtime/source and malformed input arrays."""

    monkeypatch.setattr(module, "_jl", None)
    with pytest.raises(ImportError, match="juliacall"):
        module._ensure_loaded()

    class MissingPath:
        def __truediv__(self, _name: str) -> "MissingPath":
            return self

        def is_file(self) -> bool:
            return False

    monkeypatch.setattr(module, "_jl", object())
    monkeypatch.setattr(module, "_LOADED", False)
    monkeypatch.setattr(module, "_KERNEL_DIR", MissingPath())
    with pytest.raises(FileNotFoundError, match="missing"):
        module._ensure_loaded()
    with pytest.raises(ValueError, match="one-dimensional"):
        module._as_input([[0.0]])
    with pytest.raises(ValueError, match="finite"):
        module._as_input([math.nan])


def test_julia_alpha_rejects_mismatched_drive_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Julia alpha facade validates paired drive lengths before allocation."""

    monkeypatch.setattr(julia_alpha, "_ensure_loaded", lambda: object())
    with pytest.raises(ValueError, match="inh_current length mismatch"):
        julia_alpha.simulate_alpha(*([0.0] * 11), [], [0.0])


@pytest.mark.parametrize(
    ("module", "function_name", "native_name", "array_count"),
    [
        (julia_adaptive, "simulate_adaptive_threshold_if", "simulate_adaptive_threshold_if_b", 1),
        (julia_alpha, "simulate_alpha", "simulate_alpha_b", 2),
    ],
)
@pytest.mark.parametrize("failure_kind", ("foreign", "configuration", "unknown_julia"))
def test_julia_model_facades_translate_only_classified_failures(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    function_name: str,
    native_name: str,
    array_count: int,
    failure_kind: str,
) -> None:
    """Julia errors translate only when the runtime classifies their category."""

    class FakeJuliaModule:
        def __getattr__(self, name: str) -> object:
            if name == native_name:
                return lambda *_args: (_ for _ in ()).throw(RuntimeError("native failure"))
            if name == "is_configuration_error":
                return lambda _exc: failure_kind == "configuration"
            if name == "is_candidate_error":
                return lambda _exc: False
            raise AttributeError(name)

    monkeypatch.setattr(module, "_ensure_loaded", lambda: FakeJuliaModule())
    monkeypatch.setattr(module, "is_julia_error", lambda _exc: failure_kind != "foreign")
    function = getattr(module, function_name)
    scalar_count = len(inspect.signature(function).parameters) - array_count
    arguments = [0.0] * scalar_count + [np.empty(0)] * array_count
    expected = ValueError if failure_kind == "configuration" else RuntimeError

    with pytest.raises(expected, match="native failure"):
        function(*arguments)


def test_gpu_runtime_failure_is_latched_and_warned(monkeypatch: pytest.MonkeyPatch) -> None:
    """One CUDA runtime failure permanently selects the safe CPU path."""

    monkeypatch.setattr(gpu_backend, "_GPU_RUNTIME_BROKEN", False)
    with pytest.warns(UserWarning, match="falling back to NumPy"):
        gpu_backend._mark_gpu_runtime_broken(RuntimeError("CUDA unavailable"))
    assert gpu_backend._GPU_RUNTIME_BROKEN is True
