# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused accelerator loader and result-contract coverage

"""Close fail-closed loader, native-dispatch, and result-validation branches."""

from __future__ import annotations

import importlib
import inspect
import math
import sys
from types import ModuleType

import numpy as np
import pytest

from sc_neurocore.accel import escape_rate, sigmoid_rate, threshold_linear_rate, wilson_cowan

RATE_MODULES = (escape_rate, sigmoid_rate, threshold_linear_rate)


@pytest.mark.parametrize(
    "module_name",
    [
        "sc_neurocore.accel.escape_rate",
        "sc_neurocore.accel.sigmoid_rate",
        "sc_neurocore.accel.threshold_linear_rate",
        "sc_neurocore.accel.wilson_cowan",
    ],
)
def test_accelerator_import_falls_back_when_rust_engine_is_absent(
    monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    """Module initialisation records a deterministic Python-only fallback."""

    original_import = importlib.import_module

    def import_without_engine(name: str, package: str | None = None) -> ModuleType:
        if name == "sc_neurocore_engine":
            raise ImportError("engine unavailable")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_engine)

    module = importlib.reload(sys.modules[module_name])

    assert module._HAS_RUST is False
    assert module._engine_simulate is None
    monkeypatch.undo()
    importlib.reload(module)


@pytest.mark.parametrize("module", RATE_MODULES)
def test_julia_loader_fails_closed_at_each_external_boundary(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Missing runtime, missing source, and load errors all remain unavailable."""

    monkeypatch.setattr(module, "_julia_module", None)
    monkeypatch.setattr(module.importlib_util, "find_spec", lambda _name: None)
    assert module.ensure_julia_loaded() is False

    monkeypatch.setattr(module.importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(module.os.path, "isfile", lambda _path: False)
    assert module.ensure_julia_loaded() is False

    monkeypatch.setattr(module.os.path, "isfile", lambda _path: True)

    def fail_import(_name: str) -> ModuleType:
        raise RuntimeError("Julia load failed")

    monkeypatch.setattr(module.importlib, "import_module", fail_import)
    assert module.ensure_julia_loaded() is False


@pytest.mark.parametrize(
    ("module", "library_field", "loader_name"),
    [
        (module, field, loader)
        for module in RATE_MODULES
        for field, loader in (("_go_lib", "ensure_go_loaded"), ("_mojo_lib", "ensure_mojo_loaded"))
    ],
)
def test_c_library_loader_rejects_missing_files_and_loader_errors(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    library_field: str,
    loader_name: str,
) -> None:
    """Go and Mojo loaders fail closed before exposing a partial library."""

    monkeypatch.setattr(module, library_field, None)
    monkeypatch.setattr(module.os.path, "isfile", lambda _path: False)
    assert getattr(module, loader_name)() is False

    monkeypatch.setattr(module.os.path, "isfile", lambda _path: True)

    def fail_cdll(_path: str) -> object:
        raise OSError("shared object rejected")

    monkeypatch.setattr(module.ctypes, "CDLL", fail_cdll)
    assert getattr(module, loader_name)() is False


@pytest.mark.parametrize("module", RATE_MODULES)
def test_c_library_configuration_requires_the_expected_symbol(module: ModuleType) -> None:
    """A loadable object without its governed C symbol is not a backend."""

    assert module._configure_c_library(object(), mojo=False) is None


@pytest.mark.parametrize("module", (sigmoid_rate, threshold_linear_rate))
def test_rate_backend_inventory_rejects_unknown_names(module: ModuleType) -> None:
    """Rate dispatch accepts Python and rejects undeclared backend names."""

    assert module.backend_available("python") is True
    assert module.backend_available("unknown") is False


@pytest.mark.parametrize(
    ("module", "function_name", "state_name"),
    [
        (module, function_name, state_name)
        for module in RATE_MODULES
        for function_name, state_name in (
            ("simulate_rust", "_engine_simulate"),
            ("simulate_julia", "_julia_module"),
            ("simulate_go", "_go_lib"),
            ("simulate_mojo", "_mojo_lib"),
        )
    ],
)
def test_native_dispatch_rejects_unavailable_backend(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    function_name: str,
    state_name: str,
) -> None:
    """Every direct native facade checks availability before calling outward."""

    monkeypatch.setattr(module, state_name, None)
    function = getattr(module, function_name)
    arguments = [0 for _ in inspect.signature(function).parameters]

    with pytest.raises(RuntimeError, match="unavailable"):
        function(*arguments)


@pytest.mark.parametrize("module", (sigmoid_rate, threshold_linear_rate))
def test_scalar_rate_normaliser_rejects_non_numeric_state(module: ModuleType) -> None:
    """Scalar rate results reject objects that cannot enter the numeric domain."""

    with pytest.raises(FloatingPointError, match="non-numeric"):
        module.normalise_result([object()], object(), n_steps=1, initial_rate=0.0)


@pytest.mark.parametrize(
    ("trace", "final", "match"),
    [
        ([0.0, 0.0], 0.0, "malformed"),
        ([-1.0], 0.0, "negative rate"),
        ([0.0], -1.0, "invalid final rate"),
        ([0.0], 1.0, "disagrees"),
    ],
)
def test_threshold_linear_normaliser_rejects_non_atomic_results(
    trace: list[float], final: float, match: str
) -> None:
    """Threshold-linear traces and final receipts must agree and stay nonnegative."""

    with pytest.raises(FloatingPointError, match=match):
        threshold_linear_rate.normalise_result(trace, final, n_steps=1, initial_rate=0.0)


def test_escape_result_normaliser_rejects_each_malformed_state_class() -> None:
    """EscapeRate validates numeric type, shape, finiteness, and binary events."""

    invalid_cases = (
        (object(), (), 0.0, 1, "non-numeric"),
        ([0.0], [], 0.0, 1, "malformed"),
        ([np.nan], [0.0], 0.0, 1, "non-finite"),
        ([0.0], [2.0], 0.0, 1, "non-binary"),
    )
    for trace, events, final_v, final_rng, match in invalid_cases:
        with pytest.raises(FloatingPointError, match=match):
            escape_rate._normalise_result(trace, events, final_v, final_rng)


class _RejectingLibrary:
    """Minimal C-ABI double returning a configured failure status."""

    def escape_rate_simulate_c(self, *_args: object) -> int:
        return -1

    def sigmoid_rate_simulate_c(self, *_args: object) -> int:
        return 1

    def threshold_linear_rate_simulate_c(self, *_args: object) -> int:
        return 1


def test_escape_c_boundary_rejects_kernel_status_and_spike_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The C boundary rejects both a negative status and non-atomic spike count."""

    with pytest.raises(FloatingPointError, match="kernel rejected"):
        escape_rate._simulate_c(_RejectingLibrary(), (0.0,) * 9, 1, 0, 0.0, mojo=False)

    class SpikeLibrary:
        def escape_rate_simulate_c(self, *_args: object) -> int:
            return 1

    monkeypatch.setattr(
        escape_rate,
        "_normalise_result",
        lambda *_args: (np.empty(0), np.empty(0, dtype=np.uint8), 0.0, 1),
    )
    with pytest.raises(FloatingPointError, match="spike count disagrees"):
        escape_rate._simulate_c(SpikeLibrary(), (0.0,) * 9, 1, 0, 0.0, mojo=False)


def test_scalar_rate_c_boundaries_reject_nonzero_status() -> None:
    """Both scalar-rate C ABIs reject a nonzero kernel status atomically."""

    with pytest.raises(FloatingPointError, match="kernel rejected"):
        sigmoid_rate._simulate_c(_RejectingLibrary(), 0.0, 1.0, 1.0, 0.0, 0.1, 0, 0.0, mojo=False)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        threshold_linear_rate._simulate_c(_RejectingLibrary(), 0.0, 0.0, 1.0, 0, 0.0, mojo=False)


def test_wilson_cowan_loader_and_result_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wilson-Cowan rejects unavailable lanes and every malformed result class."""

    assert wilson_cowan._logistic(1.0) > 0.5

    def fail_import(_name: str) -> ModuleType:
        raise ImportError("runtime unavailable")

    monkeypatch.setattr(wilson_cowan.importlib, "import_module", fail_import)
    assert wilson_cowan.backend_available("julia") is False
    assert wilson_cowan.backend_available("go") is False
    assert wilson_cowan.backend_available("python") is True
    assert wilson_cowan.backend_available("unknown") is False

    with pytest.raises(FloatingPointError, match="non-numeric final"):
        wilson_cowan._float(object(), "final rate")
    with pytest.raises(FloatingPointError, match="non-finite final"):
        wilson_cowan._float(math.nan, "final rate")
    with pytest.raises(FloatingPointError, match="non-numeric traces"):
        wilson_cowan.normalise_result(
            [object()], [0.0], 0.0, 0.0, n_steps=1, initial_e=0.0, initial_i=0.0, a=1.0, theta=0.0
        )
    with pytest.raises(FloatingPointError, match="out-of-range I"):
        wilson_cowan.normalise_result(
            [0.0], [2.0], 0.0, 2.0, n_steps=1, initial_e=0.0, initial_i=0.0, a=1.0, theta=0.0
        )
    with pytest.raises(FloatingPointError, match="invalid final rates"):
        wilson_cowan.normalise_result(
            [], [], 2.0, 0.0, n_steps=0, initial_e=2.0, initial_i=0.0, a=1.0, theta=0.0
        )
    with pytest.raises(FloatingPointError, match="incomplete"):
        wilson_cowan._normalise_mapping(
            {}, n_steps=0, initial_e=0.0, initial_i=0.0, a=1.0, theta=0.0
        )

    monkeypatch.setattr(wilson_cowan, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        wilson_cowan.simulate_rust(*([0.0] * 13))
