# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fidelity accelerator boundary coverage

"""Exercise the fail-closed contracts shared by the fidelity accelerators."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np
import pytest


SOURCE_MODULE_NAMES = (
    "benda_herz",
    "energy_lif",
    "mat",
    "mckean",
    "non_resetting_lif",
    "sc_non_resetting_adaptive_lif",
    "sc_normalized_energy_lif",
    "sc_resetting_mat",
    "sc_sigma_delta_accumulator",
    "sc_stochastic_rate_adaptation",
    "sigma_delta",
)
SOURCE_MODULES = tuple(
    importlib.import_module(f"sc_neurocore.accel.{name}") for name in SOURCE_MODULE_NAMES
)
MAP_MODULE_NAMES = (
    "aihara_map",
    "nagumo_sato_map",
    "sc_adaptive_threshold_map",
    "sc_chaotic_map",
)
MAP_MODULES = tuple(
    importlib.import_module(f"sc_neurocore.accel.{name}") for name in MAP_MODULE_NAMES
)


@pytest.mark.parametrize("module", SOURCE_MODULES)
def test_fidelity_dispatch_availability_fails_closed(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Missing Rust, Julia, and C-ABI modules remain explicit unavailable lanes."""

    def missing_runtime(_name: str) -> ModuleType:
        raise ImportError("runtime absent")

    monkeypatch.setattr(module.importlib, "import_module", missing_runtime)
    rust_loader = getattr(module, "_rust_module", getattr(module, "_rust", None))
    assert rust_loader() is None
    assert module.backend_available("rust") is False
    assert module.backend_available("julia") is False
    assert module.backend_available("go") is False
    assert module.backend_available("unknown") is False


@pytest.mark.parametrize("module", SOURCE_MODULES)
def test_fidelity_dispatch_auto_backend_retains_python_floor(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """An unavailable measured lane cannot remove the governed Python floor."""

    monkeypatch.setattr(module, "select_backend_order", lambda *_args, **_kwargs: ("missing",))
    monkeypatch.setattr(module, "backend_available", lambda _backend: False)
    assert module.auto_backend() == "python"


PUBLIC_FUNCTIONS = {
    name: getattr(module, next(item for item in module.__all__ if item.startswith("simulate_")))
    for name, module in zip(SOURCE_MODULE_NAMES, SOURCE_MODULES, strict=True)
}


@pytest.mark.parametrize(
    ("name", "module"), tuple(zip(SOURCE_MODULE_NAMES, SOURCE_MODULES, strict=True))
)
def test_fidelity_dispatch_rejects_bad_input_and_backend_state(
    monkeypatch: pytest.MonkeyPatch, name: str, module: ModuleType
) -> None:
    """Public dispatch rejects malformed drive, unknown lanes, and unavailable lanes."""

    function = PUBLIC_FUNCTIONS[name]
    invalid_inputs: tuple[object, ...]
    if name == "sc_stochastic_rate_adaptation":
        invalid_inputs = ([0.0], [1.0])
    else:
        invalid_inputs = ([[0.0]],)
    with pytest.raises(ValueError):
        function(*invalid_inputs)

    empty_inputs: tuple[object, ...]
    if name == "sc_stochastic_rate_adaptation":
        empty_inputs = ([], [])
    else:
        empty_inputs = ([],)
    with pytest.raises(ValueError, match="unknown"):
        function(*empty_inputs, backend="undeclared")

    monkeypatch.setattr(module, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        function(*empty_inputs, backend="go")


@pytest.mark.parametrize(
    ("name", "module"), tuple(zip(SOURCE_MODULE_NAMES, SOURCE_MODULES, strict=True))
)
def test_fidelity_dispatch_rechecks_rust_handle_atomically(
    monkeypatch: pytest.MonkeyPatch, name: str, module: ModuleType
) -> None:
    """Rust dispatch rechecks the extension immediately before its foreign call."""

    function = PUBLIC_FUNCTIONS[name]
    empty_inputs: tuple[object, ...] = (
        ([], []) if name == "sc_stochastic_rate_adaptation" else ([],)
    )
    monkeypatch.setattr(module, "backend_available", lambda _backend: True)
    rust_loader_name = "_rust" if name == "sc_stochastic_rate_adaptation" else "_rust_module"
    monkeypatch.setattr(module, rust_loader_name, lambda: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        function(*empty_inputs, backend="rust")


NORMALISER_CASES = (
    (
        "energy_lif",
        {"voltages": [0.0], "epsilon": [1.0], "events": [0], "v_final": 0.0, "epsilon_final": 1.0},
        (0.0, 1.0),
        "voltages",
        "v_final",
    ),
    (
        "mat",
        {
            "voltages": [0.0],
            "theta1": [0.0],
            "theta2": [0.0],
            "refractory": [0.0],
            "events": [0],
            "v_final": 0.0,
            "theta1_final": 0.0,
            "theta2_final": 0.0,
            "refractory_final": 0.0,
        },
        (0.0, 0.0, 0.0, 0.0),
        "voltages",
        "v_final",
    ),
    (
        "mckean",
        {"voltages": [0.0], "recovery": [0.0], "events": [0], "v_final": 0.0, "w_final": 0.0},
        (0.0, 0.0),
        "voltages",
        "v_final",
    ),
    (
        "non_resetting_lif",
        {
            "voltages": [0.0],
            "theta": [0.0],
            "refractory": [0.0],
            "events": [0],
            "v_final": 0.0,
            "theta_final": 0.0,
            "refractory_final": 0.0,
        },
        (0.0, 0.0, 0.0),
        "voltages",
        "v_final",
    ),
    (
        "sc_non_resetting_adaptive_lif",
        {"voltages": [0.0], "theta": [0.0], "events": [0], "v_final": 0.0, "theta_final": 0.0},
        (0.0, 0.0),
        "voltages",
        "v_final",
    ),
    (
        "sc_normalized_energy_lif",
        {"voltages": [0.0], "epsilon": [1.0], "events": [0], "v_final": 0.0, "epsilon_final": 1.0},
        (0.0, 1.0),
        "voltages",
        "v_final",
    ),
    (
        "sc_resetting_mat",
        {
            "voltages": [0.0],
            "theta1": [0.0],
            "theta2": [0.0],
            "events": [0],
            "v_final": 0.0,
            "theta1_final": 0.0,
            "theta2_final": 0.0,
        },
        (0.0, 0.0, 0.0),
        "voltages",
        "v_final",
    ),
    (
        "sigma_delta",
        {
            "sigma": [0.0],
            "reconstruction": [0.0],
            "events": [0],
            "sigma_final": 0.0,
            "reconstruction_final": 0.0,
        },
        (0.0, 0.0),
        "sigma",
        "sigma_final",
    ),
)


@pytest.mark.parametrize(("name", "result", "initial", "trace_key", "final_key"), NORMALISER_CASES)
def test_fidelity_normalisers_reject_malformed_receipts(
    name: str,
    result: dict[str, object],
    initial: tuple[float, ...],
    trace_key: str,
    final_key: str,
) -> None:
    """Trace shape, event alphabet, and final-state receipts are all atomic."""

    module = importlib.import_module(f"sc_neurocore.accel.{name}")
    malformed_trace = dict(result)
    malformed_trace[trace_key] = [0.0, 0.0]
    with pytest.raises(FloatingPointError, match="malformed"):
        module._normalise(malformed_trace, 1, initial)

    malformed_events = dict(result)
    malformed_events["events"] = [2]
    with pytest.raises(FloatingPointError, match="malformed"):
        module._normalise(malformed_events, 1, initial)

    malformed_final = dict(result)
    malformed_final[final_key] = np.nan
    with pytest.raises(FloatingPointError, match="disagrees"):
        module._normalise(malformed_final, 1, initial)

    if name == "mat":
        negative_refractory = dict(result)
        negative_refractory["refractory"] = [-1.0]
        negative_refractory["refractory_final"] = -1.0
        with pytest.raises(FloatingPointError, match="negative"):
            module._normalise(negative_refractory, 1, initial)


@pytest.mark.parametrize(
    ("name", "result"),
    (
        (
            "sc_sigma_delta_accumulator",
            {"sigma": [0.0, 0.0], "events": [0], "sigma_final": 0.0},
        ),
        (
            "sc_stochastic_rate_adaptation",
            {"adaptation": [0.0, 0.0], "events": [0], "a_final": 0.0},
        ),
    ),
)
def test_single_state_dispatchers_reject_malformed_native_receipts(
    monkeypatch: pytest.MonkeyPatch, name: str, result: dict[str, object]
) -> None:
    """Single-state fidelity wrappers validate native shape before publication."""

    module = importlib.import_module(f"sc_neurocore.accel.{name}")
    function = PUBLIC_FUNCTIONS[name]
    monkeypatch.setattr(module, "backend_available", lambda _backend: True)
    monkeypatch.setattr(
        module,
        "_native_module" if name != "sc_stochastic_rate_adaptation" else "_native",
        lambda _backend: type(
            "Native", (), {function.__name__: staticmethod(lambda *_args: result)}
        )(),
    )
    inputs: tuple[object, ...] = (
        ([0.0], [0.0]) if name == "sc_stochastic_rate_adaptation" else ([0.0],)
    )
    with pytest.raises(FloatingPointError, match="malformed"):
        function(*inputs, backend="go")

    if name == "sc_stochastic_rate_adaptation":
        result["adaptation"] = [0.0]
        result["a_final"] = 1.0
        with pytest.raises(FloatingPointError, match="disagrees"):
            function(*inputs, backend="go")


SIMPLE_NATIVE_CASES = tuple(
    (language, model, current_index)
    for language in ("go", "mojo")
    for model, current_index in (
        ("energy_lif", 16),
        ("mat", 13),
        ("non_resetting_lif", 10),
        ("sc_non_resetting_adaptive_lif", 9),
        ("sc_normalized_energy_lif", 11),
        ("sc_resetting_mat", 13),
        ("sc_sigma_delta_accumulator", 2),
        ("sigma_delta", 5),
    )
)


@pytest.mark.parametrize(("language", "model", "current_index"), SIMPLE_NATIVE_CASES)
def test_simple_c_abi_facades_reject_unavailable_bad_shape_and_status(
    monkeypatch: pytest.MonkeyPatch, language: str, model: str, current_index: int
) -> None:
    """Each simple C facade enforces availability, drive rank, and status zero."""

    module = importlib.import_module(f"sc_neurocore.accel.{language}.{model}")
    function = getattr(module, f"simulate_{model}")
    arguments: list[object] = [0.0] * current_index + [[]]
    monkeypatch.setattr(module, "_function", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        function(*arguments)

    monkeypatch.setattr(module, "_function", lambda *_args: 0)
    arguments[current_index] = [[0.0]]
    if not (language == "mojo" and model in {"sigma_delta", "sc_sigma_delta_accumulator"}):
        with pytest.raises(ValueError, match="one-dimensional"):
            function(*arguments)

    arguments[current_index] = []
    monkeypatch.setattr(module, "_function", lambda *_args: 1)
    with pytest.raises(FloatingPointError, match="status"):
        function(*arguments)


COMPLEX_NATIVE_CASES = tuple(
    (language, model)
    for language in ("go", "mojo")
    for model in ("amari_field", "brunel_wang", "compte_wm")
)


@pytest.mark.parametrize(("language", "model"), COMPLEX_NATIVE_CASES)
def test_vector_c_abi_facades_reject_unavailable_and_nonzero_status(
    monkeypatch: pytest.MonkeyPatch, language: str, model: str
) -> None:
    """Vector C facades fail closed before returning any partially written trace."""

    module = importlib.import_module(f"sc_neurocore.accel.{language}.{model}")
    function = getattr(module, f"simulate_{model}")
    state_name = "_FUNCTION" if model == "amari_field" else "_function"
    if model == "amari_field":
        arguments: list[Any] = [np.zeros(2), *([1.0] * 7), np.empty((0, 2))]
    elif model == "brunel_wang":
        arguments = [0.0] * 17 + [[]] * 4
    else:
        arguments = [0.0] * 24 + [[]] * 4
    monkeypatch.setattr(module, state_name, None)
    with pytest.raises(RuntimeError, match="unavailable"):
        function(*arguments)
    monkeypatch.setattr(module, state_name, lambda *_args: 1)
    with pytest.raises(FloatingPointError, match="status"):
        function(*arguments)


@pytest.mark.parametrize(
    ("language", "model"),
    tuple(
        (language, model)
        for language in ("go", "mojo")
        for model in (
            "amari_field",
            "benda_herz",
            "brunel_wang",
            "compte_wm",
            "energy_lif",
            "mat",
            "mckean",
            "non_resetting_lif",
            "sc_non_resetting_adaptive_lif",
            "sc_normalized_energy_lif",
            "sc_resetting_mat",
            "sc_sigma_delta_accumulator",
            "sc_stochastic_rate_adaptation",
            "sigma_delta",
        )
    ),
)
def test_c_abi_module_import_records_missing_library(
    monkeypatch: pytest.MonkeyPatch, language: str, model: str
) -> None:
    """Facade import converts a missing shared object into an unavailable marker."""

    module = importlib.import_module(f"sc_neurocore.accel.{language}.{model}")
    original_cdll = module.ctypes.CDLL
    monkeypatch.setattr(
        module.ctypes, "CDLL", lambda _path: (_ for _ in ()).throw(OSError("missing"))
    )
    reloaded = importlib.reload(module)
    marker = f"_HAS_{language.upper()}_{model.upper()}"
    assert getattr(reloaded, marker) is False
    monkeypatch.setattr(module.ctypes, "CDLL", original_cdll)
    importlib.reload(module)


@pytest.mark.parametrize("language", ("go", "mojo"))
def test_mckean_step_facades_reject_unavailable_and_failed_transition(
    monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    """Step-oriented McKean C facades never publish a failed transition."""

    module = importlib.import_module(f"sc_neurocore.accel.{language}.mckean")
    function = module.simulate_mckean
    state_name = "_LIB" if language == "go" else "_fn"
    monkeypatch.setattr(module, state_name, None)
    with pytest.raises(RuntimeError, match="unavailable"):
        function(*([0.0] * 7), [])

    if language == "go":
        failed = type("Failed", (), {"status": 1})()
        library = type("Library", (), {"mckean_step": staticmethod(lambda *_args: failed)})()
        monkeypatch.setattr(module, state_name, library)
        with pytest.raises(ValueError, match="transition failed"):
            function(*([0.0] * 7), [0.0])
    else:
        monkeypatch.setattr(module, state_name, lambda *_args: 1)
        with pytest.raises(FloatingPointError, match="status"):
            function(*([0.0] * 7), [])


@pytest.mark.parametrize("module", MAP_MODULES)
def test_map_dispatch_import_and_availability_fail_closed(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Map dispatch records missing engines and classifies Julia absence narrowly."""

    original_import = module.importlib.import_module

    def missing_engine(name: str) -> ModuleType:
        if name == "sc_neurocore_engine":
            raise ImportError("engine absent")
        return original_import(name)

    monkeypatch.setattr(module.importlib, "import_module", missing_engine)
    reloaded = importlib.reload(module)
    assert reloaded._HAS_RUST is False
    assert reloaded._engine_simulate is None
    monkeypatch.undo()
    module = importlib.reload(reloaded)
    assert module.backend_available("python") is True

    monkeypatch.setattr(
        module, "_ensure_julia_loaded", lambda: (_ for _ in ()).throw(ImportError())
    ) if hasattr(module, "_ensure_julia_loaded") else monkeypatch.setattr(
        module.importlib, "import_module", lambda _name: (_ for _ in ()).throw(ImportError())
    )
    assert module.backend_available("julia") is False


@pytest.mark.parametrize("module", MAP_MODULES)
def test_map_dispatch_handles_julia_and_native_loader_failures(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Julia runtime failures are classified while foreign programmer errors propagate."""

    class JuliaError(Exception):
        pass

    julia_loader_name = "_ensure_julia_loaded" if hasattr(module, "_ensure_julia_loaded") else None
    if julia_loader_name is not None:
        monkeypatch.setattr(module, julia_loader_name, lambda: (_ for _ in ()).throw(JuliaError()))
    else:
        fake = type(
            "JuliaFacade",
            (),
            {"_ensure_loaded": staticmethod(lambda: (_ for _ in ()).throw(JuliaError()))},
        )()
        monkeypatch.setattr(module.importlib, "import_module", lambda _name: fake)
    assert module.backend_available("julia") is False

    def fail_native(_backend: str) -> ModuleType:
        raise ImportError("native facade absent")

    monkeypatch.setattr(module, "_native_module", fail_native)
    assert module.backend_available("go") is False
    assert module.backend_available("unknown") is False


@pytest.mark.parametrize("module", MAP_MODULES)
def test_map_dispatch_propagates_unclassified_julia_failure(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Unexpected Julia integration defects remain visible instead of becoming absence."""

    if hasattr(module, "_ensure_julia_loaded"):
        monkeypatch.setattr(
            module, "_ensure_julia_loaded", lambda: (_ for _ in ()).throw(ZeroDivisionError())
        )
    else:
        fake = type(
            "JuliaFacade",
            (),
            {"_ensure_loaded": staticmethod(lambda: (_ for _ in ()).throw(ZeroDivisionError()))},
        )()
        monkeypatch.setattr(module.importlib, "import_module", lambda _name: fake)
    with pytest.raises(ZeroDivisionError):
        module.backend_available("julia")


@pytest.mark.parametrize("module", MAP_MODULES)
def test_map_input_and_selection_boundaries(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Map batches enforce rank, native-size, finiteness, and declared backends."""

    with pytest.raises(ValueError, match="one-dimensional"):
        module._input([[0.0]])
    monkeypatch.setattr(module, "_MAX_NATIVE_STEPS", 0)
    with pytest.raises(ValueError, match="step limit"):
        module._input([0.0])
    monkeypatch.setattr(module, "_MAX_NATIVE_STEPS", (1 << 31) - 1)
    with pytest.raises(ValueError, match="finite"):
        module._input([np.nan])

    monkeypatch.setattr(module, "select_backend_order", lambda *_args, **_kwargs: ("missing",))
    monkeypatch.setattr(module, "backend_available", lambda _backend: False)
    assert module.auto_backend() == "python"
    function = getattr(
        module,
        next(
            item
            for item in module.__all__
            if item.startswith("simulate_") and item != "simulate_python"
        ),
    )
    with pytest.raises(ValueError, match="unknown"):
        function(backend="undeclared")
    with pytest.raises(RuntimeError, match="unavailable"):
        function(backend="go")
    monkeypatch.setattr(module, "_engine_simulate", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        module._native_runner("rust")


MAP_RESULTS = {
    "aihara_map": (
        {"y": [0.0], "x": [0.5], "spikes": [1.0], "y_final": 0.0, "x_final": 0.5, "spike_count": 1},
        {"n_steps": 1, "initial_y": 0.0, "epsilon": 1.0},
    ),
    "nagumo_sato_map": (
        {"y": [0.0], "x": [1.0], "spikes": [1.0], "y_final": 0.0, "x_final": 1.0, "spike_count": 1},
        {"n_steps": 1, "initial_y": 0.0},
    ),
    "sc_adaptive_threshold_map": (
        {
            "x": [0.0],
            "theta": [0.0],
            "spikes": [0.0],
            "x_final": 0.0,
            "theta_final": 0.0,
            "spike_count": 0,
        },
        {"n_steps": 1, "initial_x": 0.0, "initial_theta": 0.0, "threshold": 0.8},
    ),
    "sc_chaotic_map": (
        {"x": [0.0], "y": [0.0], "spikes": [0.0], "x_final": 0.0, "y_final": 0.0, "spike_count": 0},
        {"n_steps": 1, "initial_x": 0.0, "initial_y": 0.0, "threshold": 0.5},
    ),
}


@pytest.mark.parametrize("name", MAP_MODULE_NAMES)
def test_map_receipts_reject_every_invalid_state_class(name: str) -> None:
    """Map receipts validate trace conversion, shape, domain, semantics, and finals."""

    module = importlib.import_module(f"sc_neurocore.accel.{name}")
    result, keywords = MAP_RESULTS[name]

    invalid = dict(result)
    invalid.pop(next(iter(invalid)))
    with pytest.raises(FloatingPointError, match="invalid"):
        module.normalise_result(invalid, **keywords)

    trace_key = "y" if name in {"aihara_map", "nagumo_sato_map"} else "x"
    malformed = dict(result)
    malformed[trace_key] = [0.0, 0.0]
    with pytest.raises(FloatingPointError, match="malformed"):
        module.normalise_result(malformed, **keywords)

    if name == "aihara_map":
        nonfinite = dict(result)
        nonfinite["y"] = [np.nan]
        with pytest.raises(FloatingPointError, match="non-finite"):
            module.normalise_result(nonfinite, **keywords)
        out_of_range = dict(result)
        out_of_range["x"] = [2.0]
        with pytest.raises(FloatingPointError, match="out-of-range"):
            module.normalise_result(out_of_range, **keywords)
        nonbinary = dict(result)
        nonbinary["spikes"] = [0.5]
        with pytest.raises(FloatingPointError, match="non-binary"):
            module.normalise_result(nonbinary, **keywords)
        wrong_readout = dict(result)
        wrong_readout["x"] = [0.6]
        with pytest.raises(FloatingPointError, match="logistic"):
            module.normalise_result(wrong_readout, **keywords)
        wrong_event = dict(result)
        wrong_event["spikes"] = [0.0]
        with pytest.raises(FloatingPointError, match="waveform"):
            module.normalise_result(wrong_event, **keywords)
    elif name == "nagumo_sato_map":
        wrong_output = dict(result)
        wrong_output["x"] = [0.0]
        with pytest.raises(FloatingPointError, match="disagrees"):
            module.normalise_result(wrong_output, **keywords)
    else:
        clamped = dict(result)
        clamped[trace_key] = [20.0]
        with pytest.raises(FloatingPointError, match="clamp"):
            module.normalise_result(clamped, **keywords)
        nonbinary = dict(result)
        nonbinary["spikes"] = [0.5]
        with pytest.raises(FloatingPointError, match="non-binary"):
            module.normalise_result(nonbinary, **keywords)
        wrong_event = dict(result)
        wrong_event["spikes"] = [1.0]
        with pytest.raises(FloatingPointError, match="crossing"):
            module.normalise_result(wrong_event, **keywords)

    final_key = "y_final" if name in {"aihara_map", "nagumo_sato_map"} else "x_final"
    invalid_final = dict(result)
    invalid_final[final_key] = object()
    with pytest.raises(FloatingPointError, match="invalid"):
        module.normalise_result(invalid_final, **keywords)
    wrong_final = dict(result)
    wrong_final[final_key] = 2.0
    with pytest.raises(FloatingPointError, match="disagrees"):
        module.normalise_result(wrong_final, **keywords)
    invalid_count = dict(result)
    invalid_count["spike_count"] = True
    with pytest.raises(FloatingPointError, match="invalid spike_count"):
        module.normalise_result(invalid_count, **keywords)
    wrong_count = dict(result)
    wrong_count["spike_count"] = 9
    with pytest.raises(FloatingPointError, match="spike_count disagrees"):
        module.normalise_result(wrong_count, **keywords)


MAP_NATIVE_CASES = tuple(
    (language, model, scalar_count)
    for language in ("go", "mojo")
    for model, scalar_count in (
        ("aihara_map", 5),
        ("nagumo_sato_map", 4),
        ("sc_adaptive_threshold_map", 7),
        ("sc_chaotic_map", 7),
    )
)


@pytest.mark.parametrize(("language", "model", "scalar_count"), MAP_NATIVE_CASES)
def test_map_c_abi_facades_validate_input_library_and_status(
    monkeypatch: pytest.MonkeyPatch, language: str, model: str, scalar_count: int
) -> None:
    """Map C facades reject invalid input and every classified kernel status."""

    module = importlib.import_module(f"sc_neurocore.accel.{language}.{model}")
    function = getattr(module, f"simulate_{model}")
    arguments: list[object] = [0.0] * scalar_count + [[[0.0]]]
    with pytest.raises(ValueError, match="one-dimensional"):
        function(*arguments)
    monkeypatch.setattr(module, "_MAX_NATIVE_STEPS", 0)
    arguments[-1] = [0.0]
    with pytest.raises(ValueError, match="step limit"):
        function(*arguments)
    monkeypatch.setattr(module, "_MAX_NATIVE_STEPS", (1 << 31) - 1)
    arguments[-1] = [np.nan]
    with pytest.raises(ValueError, match="finite"):
        function(*arguments)
    arguments[-1] = []
    monkeypatch.setattr(module, "_lib", None)
    with pytest.raises(ImportError, match="not built"):
        function(*arguments)

    symbol = f"{model}_simulate_c"
    for status, error in (
        (2, ValueError),
        (3, ValueError),
        (4, FloatingPointError),
        (5, RuntimeError),
    ):
        library = type("Library", (), {symbol: staticmethod(lambda *_args, code=status: code)})()
        monkeypatch.setattr(module, "_lib", library)
        with pytest.raises(error):
            function(*arguments)


JULIA_MAP_MODULES = tuple(
    importlib.import_module(f"sc_neurocore.accel.julia.neurons.{name}") for name in MAP_MODULE_NAMES
)


class _MissingJuliaPath:
    """Path double that keeps joins missing."""

    def __truediv__(self, _name: str) -> "_MissingJuliaPath":
        return self

    def is_file(self) -> bool:
        return False


@pytest.mark.parametrize("module", JULIA_MAP_MODULES)
def test_julia_map_loader_requires_runtime_and_source(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Julia map loading fails explicitly when either runtime or source is absent."""

    monkeypatch.setattr(module, "_jl", None)
    with pytest.raises(ImportError, match="juliacall"):
        module._ensure_loaded()
    monkeypatch.setattr(module, "_jl", object())
    monkeypatch.setattr(module, "_LOADED", False)
    monkeypatch.setattr(module, "_KERNEL_DIR", _MissingJuliaPath())
    with pytest.raises(FileNotFoundError, match="missing"):
        module._ensure_loaded()


@pytest.mark.parametrize(
    ("name", "module"), tuple(zip(MAP_MODULE_NAMES, JULIA_MAP_MODULES, strict=True))
)
def test_julia_map_facades_validate_drive_and_translate_runtime_errors(
    monkeypatch: pytest.MonkeyPatch, name: str, module: ModuleType
) -> None:
    """Julia map facades preserve drive validation and typed exception semantics."""

    function = getattr(module, f"simulate_{name}")
    scalar_count = len(importlib.import_module("inspect").signature(function).parameters) - 1
    scalars = [0.0] * scalar_count
    with pytest.raises(ValueError):
        function(*scalars, [[0.0]])
    with pytest.raises(ValueError):
        function(*scalars, [np.nan])

    class ForeignError(Exception):
        pass

    class FakeJulia:
        def __getattr__(self, attribute: str) -> Any:
            if attribute.startswith("simulate_"):
                return lambda *_args: (_ for _ in ()).throw(ForeignError("failed"))
            if attribute == "is_configuration_error":
                return lambda _exc: False
            if attribute == "is_candidate_error":
                return lambda _exc: False
            raise AttributeError(attribute)

    monkeypatch.setattr(module, "_ensure_loaded", lambda: FakeJulia())
    monkeypatch.setattr(module, "is_julia_error", lambda _exc: False)
    with pytest.raises(ForeignError):
        function(*scalars, [])

    monkeypatch.setattr(module, "is_julia_error", lambda _exc: True)
    if name == "sc_chaotic_map":
        with pytest.raises(ValueError, match="failed"):
            function(*scalars, [])
        return

    for classification, error in (("configuration", ValueError), ("candidate", FloatingPointError)):
        fake = FakeJulia()
        fake.is_configuration_error = lambda _exc, kind=classification: kind == "configuration"
        fake.is_candidate_error = lambda _exc, kind=classification: kind == "candidate"
        monkeypatch.setattr(module, "_ensure_loaded", lambda value=fake: value)
        with pytest.raises(error, match="failed"):
            function(*scalars, [])
    monkeypatch.setattr(module, "_ensure_loaded", lambda: FakeJulia())
    with pytest.raises(ForeignError):
        function(*scalars, [])


JULIA_LOADER_CASES = (
    ("amari_field", "_ensure_amari_field_loaded", "_AMARI_FIELD_LOADED"),
    ("benda_herz", "_ensure_benda_herz_loaded", "_BENDA_HERZ_LOADED"),
    (
        "benda_herz",
        "_ensure_sc_stochastic_rate_adaptation_loaded",
        "_SC_STOCHASTIC_RATE_ADAPTATION_LOADED",
    ),
    ("brunel_wang", "_ensure_brunel_wang_loaded", "_BRUNEL_WANG_LOADED"),
    ("compte_wm", "_ensure_compte_wm_loaded", "_COMPTE_WM_LOADED"),
    ("energy_lif", "_ensure_energy_lif_loaded", "_ENERGY_LIF_LOADED"),
    (
        "energy_lif",
        "_ensure_sc_normalized_energy_lif_loaded",
        "_SC_NORMALIZED_ENERGY_LIF_LOADED",
    ),
    ("mat", "_ensure_mat_loaded", "_MAT_LOADED"),
    ("mat", "_ensure_sc_resetting_mat_loaded", "_SC_RESETTING_MAT_LOADED"),
    ("mckean", "_ensure_mckean_loaded", "_MCKEAN_LOADED"),
    ("mckean", "_ensure_sc_triangular_mckean_loaded", "_SC_TRIANGULAR_MCKEAN_LOADED"),
    (
        "non_resetting_lif",
        "_ensure_non_resetting_lif_loaded",
        "_NON_RESETTING_LIF_LOADED",
    ),
    (
        "non_resetting_lif",
        "_ensure_sc_non_resetting_adaptive_lif_loaded",
        "_SC_NON_RESETTING_ADAPTIVE_LIF_LOADED",
    ),
    ("sigma_delta", "_ensure_sigma_delta_loaded", "_SIGMA_DELTA_LOADED"),
    (
        "sigma_delta",
        "_ensure_sc_sigma_delta_accumulator_loaded",
        "_SC_SIGMA_DELTA_ACCUMULATOR_LOADED",
    ),
)


@pytest.mark.parametrize(("module_name", "loader_name", "state_name"), JULIA_LOADER_CASES)
def test_fidelity_julia_loaders_require_runtime_and_kernel_source(
    monkeypatch: pytest.MonkeyPatch, module_name: str, loader_name: str, state_name: str
) -> None:
    """Every fidelity Julia kernel is loaded only from an available source file."""

    module = importlib.import_module(f"sc_neurocore.accel.julia.neurons.{module_name}")
    loader = getattr(module, loader_name)
    monkeypatch.setattr(module, "_jl", None)
    with pytest.raises(ImportError, match="juliacall"):
        loader()
    monkeypatch.setattr(module, "_jl", object())
    monkeypatch.setattr(module, state_name, False)
    monkeypatch.setattr(module, "_KERNEL_DIR", _MissingJuliaPath())
    with pytest.raises(FileNotFoundError, match="missing"):
        loader()


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    (
        ("energy_lif", "simulate_energy_lif"),
        ("energy_lif", "simulate_sc_normalized_energy_lif"),
        ("mat", "simulate_mat"),
        ("mat", "simulate_sc_resetting_mat"),
        ("mckean", "simulate_mckean"),
        ("non_resetting_lif", "simulate_non_resetting_lif"),
        ("non_resetting_lif", "simulate_sc_non_resetting_adaptive_lif"),
        ("sigma_delta", "simulate_sigma_delta"),
        ("sigma_delta", "simulate_sc_sigma_delta_accumulator"),
    ),
)
def test_fidelity_julia_facades_reject_nonfinite_or_ranked_drive(
    module_name: str, function_name: str
) -> None:
    """Julia fidelity facades reject invalid drive before runtime loading."""

    module = importlib.import_module(f"sc_neurocore.accel.julia.neurons.{module_name}")
    function = getattr(module, function_name)
    with pytest.raises(ValueError, match="finite"):
        function([np.nan])
    with pytest.raises(ValueError, match="one-dimensional"):
        function([[0.0]])


def test_julia_amari_facade_rejects_shape_and_finiteness() -> None:
    """Amari's Julia facade validates both field state and two-dimensional drive."""

    module = importlib.import_module("sc_neurocore.accel.julia.neurons.amari_field")
    with pytest.raises(ValueError, match="shape"):
        module.simulate_amari_field(np.zeros(2), *([1.0] * 7), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        module.simulate_amari_field(np.asarray([0.0, np.nan]), *([1.0] * 7), np.empty((0, 2)))


FIELD_MODULE_NAMES = ("amari_field", "brunel_wang", "compte_wm")
FIELD_MODULES = tuple(
    importlib.import_module(f"sc_neurocore.accel.{name}") for name in FIELD_MODULE_NAMES
)


@pytest.mark.parametrize("module", FIELD_MODULES)
def test_field_dispatch_availability_and_auto_floor(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType
) -> None:
    """Field-model dispatch fails closed across engine, Julia, and native discovery."""

    monkeypatch.setattr(
        module.importlib, "import_module", lambda _name: (_ for _ in ()).throw(ImportError())
    )
    assert module._engine_class() is None
    assert module.backend_available("rust") is False
    assert module.backend_available("julia") is False
    assert module.backend_available("go") is False
    assert module.backend_available("unknown") is False
    monkeypatch.setattr(module, "select_backend_order", lambda *_args, **_kwargs: ("missing",))
    monkeypatch.setattr(module, "backend_available", lambda _backend: False)
    assert module.auto_backend() == "python"


def test_amari_input_receipt_and_rust_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Amari validates vector geometry, receipt identity, and Rust extension freshness."""

    module = importlib.import_module("sc_neurocore.accel.amari_field")
    with pytest.raises(ValueError, match="at least two"):
        module._inputs([0.0], [])
    with pytest.raises(ValueError, match="shape"):
        module._inputs([0.0, 0.0], np.empty((0, 3)))
    with pytest.raises(ValueError, match="finite"):
        module._inputs([0.0, 0.0], [[np.nan, 0.0]])

    valid = {"states": [[0.0, 0.0]], "mean_rates": [0.5], "final_state": [0.0, 0.0]}
    invalid = dict(valid)
    invalid.pop("states")
    with pytest.raises(FloatingPointError, match="invalid states"):
        module._normalise(invalid, 1, 2, np.zeros(2))
    malformed = dict(valid)
    malformed["states"] = [[np.nan, 0.0]]
    with pytest.raises(FloatingPointError, match="malformed"):
        module._normalise(malformed, 1, 2, np.zeros(2))
    bad_rate = dict(valid)
    bad_rate["mean_rates"] = [2.0]
    with pytest.raises(FloatingPointError, match=r"\[0, 1\]"):
        module._normalise(bad_rate, 1, 2, np.zeros(2))
    wrong_final = dict(valid)
    wrong_final["final_state"] = [1.0, 0.0]
    with pytest.raises(FloatingPointError, match="disagrees"):
        module._normalise(wrong_final, 1, 2, np.zeros(2))

    arguments = (np.zeros(2), *([1.0] * 7), np.empty((0, 2)))
    monkeypatch.setattr(module, "_engine_class", lambda: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        module._simulate_rust(*arguments)

    class StaleEngine:
        __text_signature__ = "(n)"

    monkeypatch.setattr(module, "_engine_class", lambda: StaleEngine)
    with pytest.raises(RuntimeError, match="stale"):
        module._simulate_rust(*arguments)

    public_state = np.linspace(-0.2, 0.2, 8)
    with pytest.raises(ValueError, match="unknown"):
        module.simulate_amari_field(public_state, currents=[], backend="undeclared")
    monkeypatch.setattr(module, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        module.simulate_amari_field(public_state, currents=[], backend="go")


BRUNEL_CONFIG = (
    -70.0,
    0.0,
    -70.0,
    -55.0,
    -50.0,
    20.0,
    2.0,
    2.08,
    0.104,
    0.327,
    1.25,
    0.0,
    0.0,
    -70.0,
    0.5,
    1.0,
    0.1,
)
COMPTE_CONFIG = (
    -70.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.025,
    0.0031,
    0.000381,
    0.001336,
    -70.0,
    0.0,
    -70.0,
    0.5,
    1.0,
    2.0,
    100.0,
    2.0,
    10.0,
    0.5,
    -50.0,
    -60.0,
    2.0,
    0.02,
)


def test_brunel_wang_input_receipt_and_selection_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brunel-Wang validates four gates, state receipts, and selected runtime."""

    module = importlib.import_module("sc_neurocore.accel.brunel_wang")
    with pytest.raises(ValueError, match="equal-length"):
        module._inputs([], [], [], [0.0])
    with pytest.raises(ValueError, match="non-negative"):
        module._inputs([-1.0], [0.0], [0.0], [0.0])
    valid = {
        "voltages": [0.0],
        "refractory": [0.0],
        "events": [0],
        "v_final": 0.0,
        "ref_final": 0.0,
    }
    malformed = dict(valid)
    malformed["voltages"] = [0.0, 0.0]
    with pytest.raises(FloatingPointError, match="malformed"):
        module._normalise(malformed, 1, (0.0, 0.0))
    bad_events = dict(valid)
    bad_events["events"] = [2]
    with pytest.raises(FloatingPointError, match="events"):
        module._normalise(bad_events, 1, (0.0, 0.0))
    bad_final = dict(valid)
    bad_final["v_final"] = np.nan
    with pytest.raises(FloatingPointError, match="invalid"):
        module._normalise(bad_final, 1, (0.0, 0.0))
    wrong_final = dict(valid)
    wrong_final["v_final"] = 1.0
    with pytest.raises(FloatingPointError, match="disagrees"):
        module._normalise(wrong_final, 1, (0.0, 0.0))
    negative_ref = dict(valid)
    negative_ref["refractory"] = [-1.0]
    negative_ref["ref_final"] = -1.0
    with pytest.raises(FloatingPointError, match="negative"):
        module._normalise(negative_ref, 1, (0.0, 0.0))
    monkeypatch.setattr(module, "_engine_class", lambda: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        module._rust_runner(BRUNEL_CONFIG, (np.empty(0),) * 4)
    arguments = (*BRUNEL_CONFIG, [], [], [], [])
    with pytest.raises(ValueError, match="unknown"):
        module.simulate_brunel_wang(*arguments, backend="undeclared")
    monkeypatch.setattr(module, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        module.simulate_brunel_wang(*arguments, backend="go")


def test_compte_input_receipt_and_selection_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compte validates event inputs, gated state receipts, and selected runtime."""

    module = importlib.import_module("sc_neurocore.accel.compte_wm")
    with pytest.raises(ValueError, match="equal-length"):
        module._inputs([], [], [], [0])
    with pytest.raises(ValueError, match="finite"):
        module._inputs([np.nan], [0], [0], [0])
    with pytest.raises(ValueError, match="zero or one"):
        module._inputs([0.0], [2], [0], [0])
    traces = {key: [0.0] for key in module._TRACE_KEYS}
    valid = {**traces, "events": [0], **{key: 0.0 for key in module._FINAL_KEYS}}
    malformed = dict(valid)
    malformed[module._TRACE_KEYS[0]] = [0.0, 0.0]
    with pytest.raises(FloatingPointError, match="malformed"):
        module._normalise(malformed, 1, (0.0,) * 6)
    bad_events = dict(valid)
    bad_events["events"] = [2]
    with pytest.raises(FloatingPointError, match="events"):
        module._normalise(bad_events, 1, (0.0,) * 6)
    bad_final = dict(valid)
    bad_final[module._FINAL_KEYS[0]] = np.nan
    with pytest.raises(FloatingPointError, match="invalid"):
        module._normalise(bad_final, 1, (0.0,) * 6)
    wrong_final = dict(valid)
    wrong_final[module._FINAL_KEYS[0]] = 1.0
    with pytest.raises(FloatingPointError, match="disagrees"):
        module._normalise(wrong_final, 1, (0.0,) * 6)
    invalid_nmda = dict(valid)
    invalid_nmda["s_nmda"] = [2.0]
    invalid_nmda["s_nmda_final"] = 2.0
    with pytest.raises(FloatingPointError, match="unit interval"):
        module._normalise(invalid_nmda, 1, (0.0,) * 6)
    negative_ref = dict(valid)
    negative_ref["refractory"] = [-1.0]
    negative_ref["ref_final"] = -1.0
    with pytest.raises(FloatingPointError, match="negative"):
        module._normalise(negative_ref, 1, (0.0,) * 6)
    monkeypatch.setattr(module, "_engine_class", lambda: None)
    with pytest.raises(RuntimeError, match="unavailable"):
        module._rust_runner(COMPTE_CONFIG, (np.empty(0),) * 4)
    arguments = (*COMPTE_CONFIG, [], [], [], [])
    with pytest.raises(ValueError, match="unknown"):
        module.simulate_compte_wm(*arguments, backend="undeclared")
    monkeypatch.setattr(module, "backend_available", lambda _backend: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        module.simulate_compte_wm(*arguments, backend="go")


def test_benda_herz_receipts_reject_shape_events_and_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benda-Herz native receipts preserve trace shape, event alphabet, and finals."""

    module = importlib.import_module("sc_neurocore.accel.benda_herz")
    result: dict[str, object] = {
        "adaptation": [0.0, 0.0],
        "phases": [0.0],
        "events": [0],
        "a_final": 0.0,
        "phase_final": 0.0,
    }
    native = type("Native", (), {"simulate_benda_herz": staticmethod(lambda *_args: result)})()
    monkeypatch.setattr(module, "backend_available", lambda _backend: True)
    monkeypatch.setattr(module, "_native_module", lambda _backend: native)
    with pytest.raises(FloatingPointError, match="adaptation"):
        module.simulate_benda_herz([0.0], backend="go")
    result["adaptation"] = [0.0]
    result["events"] = [2]
    with pytest.raises(FloatingPointError, match="events"):
        module.simulate_benda_herz([0.0], backend="go")
    result["events"] = [0]
    result["a_final"] = 1.0
    with pytest.raises(FloatingPointError, match="disagrees"):
        module.simulate_benda_herz([0.0], backend="go")


@pytest.mark.parametrize(("language", "model", "scalar_count"), MAP_NATIVE_CASES)
def test_map_c_abi_import_records_missing_library(
    monkeypatch: pytest.MonkeyPatch, language: str, model: str, scalar_count: int
) -> None:
    """Map facades expose an unavailable marker when their shared object is absent."""

    del scalar_count
    module = importlib.import_module(f"sc_neurocore.accel.{language}.{model}")
    original_cdll = module.ctypes.CDLL
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _path: (_ for _ in ()).throw(OSError()))
    assert importlib.reload(module)._lib is None
    monkeypatch.setattr(module.ctypes, "CDLL", original_cdll)
    importlib.reload(module)


@pytest.mark.parametrize("model", ("benda_herz", "sc_stochastic_rate_adaptation"))
def test_stepwise_go_fidelity_facades_execute_and_reject_transition(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """Go step facades reject absent/failed kernels and return complete traces."""

    module = importlib.import_module(f"sc_neurocore.accel.go.{model}")
    function = getattr(module, f"simulate_{model}")
    scalar_count = 7
    arrays: tuple[object, ...] = ([0.0], [0.0]) if model.startswith("sc_stochastic") else ([0.0],)
    monkeypatch.setattr(module, "_LIB", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        function(*([0.0] * scalar_count), *arrays)

    failed = type("Result", (), {"status": 1})()
    symbol = "sc_sra_step" if model.startswith("sc_stochastic") else "benda_herz_step"
    library = type("Library", (), {symbol: staticmethod(lambda *_args: failed)})()
    monkeypatch.setattr(module, "_LIB", library)
    with pytest.raises(ValueError, match="failed"):
        function(*([0.0] * scalar_count), *arrays)

    succeeded = type("Result", (), {"status": 0, "a": 0.25, "phase": 0.5, "event": 1})()
    library = type("Library", (), {symbol: staticmethod(lambda *_args: succeeded)})()
    monkeypatch.setattr(module, "_LIB", library)
    result = function(*([0.0] * scalar_count), *arrays)
    assert np.asarray(result["events"]).tolist() == [1]


@pytest.mark.parametrize("model", ("benda_herz", "sc_stochastic_rate_adaptation"))
def test_batch_mojo_fidelity_facades_execute_and_reject_status(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    """Mojo batch facades reject absent/nonzero kernels and return complete arrays."""

    module = importlib.import_module(f"sc_neurocore.accel.mojo.{model}")
    function = getattr(module, f"simulate_{model}")
    arrays: tuple[object, ...] = ([0.0], [0.0]) if model.startswith("sc_stochastic") else ([0.0],)
    arguments = (*([0.0] * 7), *arrays)
    monkeypatch.setattr(module, "_fn", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        function(*arguments)
    monkeypatch.setattr(module, "_fn", lambda *_args: 1)
    with pytest.raises(FloatingPointError, match="status"):
        function(*arguments)
    monkeypatch.setattr(module, "_fn", lambda *_args: 0)
    assert np.asarray(function(*arguments)["events"]).shape == (1,)


def test_sc_triangular_mckean_fail_closed_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retained triangular identity validates availability and every receipt class."""

    module = importlib.import_module("sc_neurocore.accel.sc_triangular_mckean")

    class MissingNeuron:
        def __init__(self, *_args: object) -> None:
            self.v = 0.0
            self.w = 0.0

        def step(self, _current: float) -> int:
            return 0

        def simulate(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("missing")

    monkeypatch.setattr(module, "SCTriangularMcKeanNeuron", MissingNeuron)
    assert module.backend_available("go") is False
    monkeypatch.setattr(module, "select_backend_order", lambda *_args, **_kwargs: ("missing",))
    monkeypatch.setattr(module, "backend_available", lambda _backend: False)
    assert module.auto_backend() == "python"
    with pytest.raises(RuntimeError, match="unavailable"):
        module.simulate_sc_triangular_mckean([], backend="go")

    result: dict[str, object] = {
        "voltages": [0.0, 0.0],
        "recovery": [0.0],
        "events": [0],
        "v_final": 0.0,
        "w_final": 0.0,
    }
    monkeypatch.setattr(module, "_run", lambda *_args: result)
    monkeypatch.setattr(module, "backend_available", lambda _backend: True)
    with pytest.raises(FloatingPointError, match="voltages"):
        module.simulate_sc_triangular_mckean([0.0], backend="go")
    result["voltages"] = [0.0]
    result["events"] = [2]
    with pytest.raises(FloatingPointError, match="events"):
        module.simulate_sc_triangular_mckean([0.0], backend="go")
    result["events"] = [0]
    result["v_final"] = 1.0
    with pytest.raises(FloatingPointError, match="disagrees"):
        module.simulate_sc_triangular_mckean([0.0], backend="go")


def test_sc_triangular_julia_loader_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SC triangular Julia loader includes its checked source exactly once."""

    module = importlib.import_module("sc_neurocore.accel.julia.neurons.mckean")

    class PresentPath(_MissingJuliaPath):
        def is_file(self) -> bool:
            return True

    runtime = type(
        "Runtime",
        (),
        {"include": staticmethod(lambda _path: None), "SCTriangularMcKeanAccel": object()},
    )()
    monkeypatch.setattr(module, "_jl", runtime)
    monkeypatch.setattr(module, "_KERNEL_DIR", PresentPath())
    monkeypatch.setattr(module, "_SC_TRIANGULAR_MCKEAN_LOADED", False)
    assert module._ensure_sc_triangular_mckean_loaded() is runtime.SCTriangularMcKeanAccel


def _mojo_network_shell() -> tuple[ModuleType, Any]:
    """Build a real-spec Mojo facade shell without invoking the native constructor."""

    module = importlib.import_module("sc_neurocore.accel.mojo.sc_compte_wm_network")
    network = object.__new__(module.SCCompteWMMojoNetwork)
    network.spec = module.SCCompteWMNetworkSpec()
    network._state = network._initial_state()
    return module, network


def test_mojo_network_import_unavailable_provenance_and_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The network facade records library absence and preserves source/runtime custody."""

    module = importlib.import_module("sc_neurocore.accel.mojo.sc_compte_wm_network")
    original_cdll = module.ctypes.CDLL
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _path: (_ for _ in ()).throw(OSError()))
    reloaded = importlib.reload(module)
    assert reloaded._LIBRARY is None
    with pytest.raises(RuntimeError, match="unavailable"):
        reloaded.SCCompteWMMojoNetwork()
    monkeypatch.setattr(module.ctypes, "CDLL", original_cdll)
    module = importlib.reload(module)
    _, network = _mojo_network_shell()
    provenance = network.provenance
    assert len(provenance.source_sha256) == len(provenance.library_sha256) == 64
    network._state.step_index = 7
    network.reset()
    assert network._state.step_index == 0


def test_mojo_network_spectrum_counter_and_input_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native preprocessing and counter failures cannot expose partial arrays."""

    module, network = _mojo_network_shell()
    module._SPECTRUM_CACHE.clear()
    failed_spectrum = type(
        "Library",
        (),
        {"sc_compte_wm_network_kernel_spectrum_c": staticmethod(lambda *_args: 1)},
    )()
    monkeypatch.setattr(module, "_LIBRARY", failed_spectrum)
    with pytest.raises(RuntimeError, match="footprint"):
        network._spectrum(1.0, 10.0)

    failed_counter = type(
        "Library",
        (),
        {"sc_compte_wm_network_counter_poisson_c": staticmethod(lambda *_args: 1)},
    )()
    monkeypatch.setattr(module, "_LIBRARY", failed_counter)
    with pytest.raises(RuntimeError, match="counter-Poisson"):
        network._counter_events(1, 0, 0)
    with pytest.raises(ValueError, match="shape"):
        network._events("events", [], 1)
    with pytest.raises(ValueError, match="non-negative"):
        network._events("events", [-1], 1)
    with pytest.raises(ValueError, match="shape"):
        network._current([0.0])


def test_mojo_network_state_validation_covers_each_safety_envelope() -> None:
    """Checkpoint validation rejects scalar, geometry, voltage, and channel defects."""

    _module, network = _mojo_network_shell()
    state = network._initial_state()
    state.step_index = True
    with pytest.raises(ValueError, match="step_index"):
        network._validate_state(state)
    state = network._initial_state()
    state.v_exc_mv = state.v_exc_mv[:-1]
    with pytest.raises(ValueError, match="shape"):
        network._validate_state(state)
    state = network._initial_state()
    state.v_exc_mv[0] = 101.0
    with pytest.raises(ValueError, match="excitatory voltage"):
        network._validate_state(state)
    state = network._initial_state()
    state.v_inh_mv[0] = 101.0
    with pytest.raises(ValueError, match="inhibitory voltage"):
        network._validate_state(state)
    state = network._initial_state()
    state.external_ampa_exc[0] = -1.0
    with pytest.raises(ValueError, match="channel state"):
        network._validate_state(state)
    state = network._initial_state()
    state.recurrent_nmda[0] = 2.0
    with pytest.raises(ValueError, match="bounded by one"):
        network._validate_state(state)


def test_mojo_network_step_stimulus_and_run_validation() -> None:
    """Step pairing, localized current, timing, windows, and epochs fail closed."""

    module, network = _mojo_network_shell()
    with pytest.raises(ValueError, match="supplied together"):
        network.step(external_exc_events=np.zeros(2048, dtype=np.int64))
    stimulus = module.SCCompteWMStimulus(0.0, 0.02, 1.0, center_deg=0.0)
    current = network._stimulus_current(0.0, (stimulus,))
    assert current.shape == (2048,)
    assert np.max(current) > 0.0
    with pytest.raises(ValueError, match="duration_ms"):
        network.run(np.nan)
    with pytest.raises(ValueError, match="integral number"):
        network.run(0.03)
    with pytest.raises(ValueError, match="statistics_window_ms"):
        network.run(0.02, statistics_window_ms=np.nan)
    with pytest.raises(ValueError, match="integral number"):
        network.run(0.02, statistics_window_ms=0.03)
    late = module.SCCompteWMStimulus(0.02, 0.02, 1.0, center_deg=0.0)
    with pytest.raises(ValueError, match="within"):
        network.run(0.02, stimuli=(late,), statistics_window_ms=0.02)
