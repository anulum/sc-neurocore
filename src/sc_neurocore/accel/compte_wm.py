# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — measured-order Compte pyramidal-cell dispatch

"""Dispatch complete Compte membrane, channel, refractory, and event traces."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
CompteWMResult: TypeAlias = dict[str, FloatArray | IntArray | float]
KERNEL = "compte_wm_pyramidal_rk2_batch"
PARITY_ATOL = {
    "python": 0.0,
    "rust": 2.0e-12,
    "julia": 2.0e-12,
    "go": 2.0e-12,
    "mojo": 2.0e-10,
}
_AUTO_BACKENDS = with_floor("python")
_TRACE_KEYS = ("voltages", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "refractory")
_FINAL_KEYS = (
    "v_final",
    "s_ampa_final",
    "s_nmda_final",
    "x_nmda_final",
    "s_gaba_final",
    "ref_final",
)


class _NativeRunner(Protocol):
    """Mapping-returning contract shared by Go and Mojo facades."""

    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _engine_class() -> type[Any] | None:
    try:
        return cast(type[Any], importlib.import_module("sc_neurocore_engine").CompteWMNeuron)
    except (ImportError, AttributeError):
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.compte_wm")


def backend_available(backend: str) -> bool:
    """Return whether one named Compte runtime is executable."""
    if backend == "python":
        return True
    if backend == "rust":
        return _engine_class() is not None
    if backend == "julia":
        try:
            importlib.import_module("sc_neurocore.accel.julia.neurons")._ensure_compte_wm_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_COMPTE_WM", False))
    return False


def auto_backend() -> str:
    """Return the first executable measured lane, with Python as floor."""
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _inputs(
    currents: npt.ArrayLike,
    recurrent_events: npt.ArrayLike,
    external_events: npt.ArrayLike,
    inhibitory_events: npt.ArrayLike,
) -> tuple[FloatArray, IntArray, IntArray, IntArray]:
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    recurrent = np.ascontiguousarray(recurrent_events, dtype=np.int64)
    external = np.ascontiguousarray(external_events, dtype=np.int64)
    inhibitory = np.ascontiguousarray(inhibitory_events, dtype=np.int64)
    events = (recurrent, external, inhibitory)
    arrays = (drive, *events)
    lengths = {value.size for value in arrays}
    if any(value.ndim != 1 for value in arrays) or len(lengths) != 1:
        raise ValueError("Compte input arrays must be one-dimensional and equal-length")
    if not np.isfinite(drive).all():
        raise ValueError("Compte current input must be finite")
    if any(not np.isin(value, (0, 1)).all() for value in events):
        raise ValueError("Compte presynaptic events must contain only zero or one")
    return drive, events[0], events[1], events[2]


def _normalise(
    result: Mapping[str, object], steps: int, initial: tuple[float, ...]
) -> CompteWMResult:
    normalised: CompteWMResult = {}
    for key in _TRACE_KEYS:
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"Compte backend returned malformed {key}")
        normalised[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (steps,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("Compte backend returned malformed output events")
    normalised["events"] = events
    for index, (key, trace_key) in enumerate(zip(_FINAL_KEYS, _TRACE_KEYS, strict=True)):
        value = float(cast(float, result[key]))
        if not np.isfinite(value):
            raise FloatingPointError(f"Compte backend returned invalid {key}")
        trace = cast(FloatArray, normalised[trace_key])
        expected = initial[index] if steps == 0 else float(trace[-1])
        if value != expected:
            raise FloatingPointError(f"Compte {key} disagrees with its trace")
        normalised[key] = value
    if np.any(cast(FloatArray, normalised["s_nmda"]) < 0.0) or np.any(
        cast(FloatArray, normalised["s_nmda"]) > 1.0
    ):
        raise FloatingPointError("Compte NMDA state left its unit interval")
    if np.any(cast(FloatArray, normalised["refractory"]) < 0.0):
        raise FloatingPointError("Compte refractory state became negative")
    return normalised


def _python_runner(
    config: tuple[float, ...], inputs: tuple[FloatArray, IntArray, IntArray, IntArray]
) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron

    state = CompteWMNeuron(
        v=config[0],
        s_ampa=config[1],
        s_nmda=config[2],
        x_nmda=config[3],
        s_gaba=config[4],
        g_l=config[6],
        g_ampa=config[7],
        g_nmda=config[8],
        g_gaba=config[9],
        e_l=config[10],
        e_exc=config[11],
        e_inh=config[12],
        c_m=config[13],
        mg=config[14],
        tau_ampa=config[15],
        tau_nmda=config[16],
        tau_x=config[17],
        tau_gaba=config[18],
        alpha_nmda=config[19],
        v_threshold=config[20],
        v_reset=config[21],
        tau_ref=config[22],
        dt=config[23],
    )
    state._ref_remaining = config[5]
    steps = inputs[0].size
    traces = {key: np.empty(steps, dtype=np.float64) for key in _TRACE_KEYS}
    output_events = np.empty(steps, dtype=np.int64)
    for index, (current, recurrent, external, inhibitory) in enumerate(zip(*inputs, strict=True)):
        output_events[index] = state.step(
            float(current),
            bool(recurrent),
            external_spike=bool(external),
            inhibitory_spike=bool(inhibitory),
        )
        current_state = state.get_state()
        for key, state_key in zip(
            _TRACE_KEYS,
            ("v", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "ref_remaining"),
            strict=True,
        ):
            traces[key][index] = current_state[state_key]
    final = state.get_state()
    return {
        **traces,
        "events": output_events,
        "v_final": final["v"],
        "s_ampa_final": final["s_ampa"],
        "s_nmda_final": final["s_nmda"],
        "x_nmda_final": final["x_nmda"],
        "s_gaba_final": final["s_gaba"],
        "ref_final": final["ref_remaining"],
    }


def _rust_runner(
    config: tuple[float, ...], inputs: tuple[FloatArray, IntArray, IntArray, IntArray]
) -> Mapping[str, object]:
    engine = _engine_class()
    if engine is None:
        raise RuntimeError("Rust Compte backend is unavailable")
    state = engine(
        v=config[0],
        s_ampa=config[1],
        s_nmda=config[2],
        x_nmda=config[3],
        s_gaba=config[4],
        ref_remaining=config[5],
        g_l=config[6],
        g_ampa=config[7],
        g_nmda=config[8],
        g_gaba=config[9],
        e_l=config[10],
        e_exc=config[11],
        e_inh=config[12],
        c_m=config[13],
        mg=config[14],
        tau_ampa=config[15],
        tau_nmda=config[16],
        tau_x=config[17],
        tau_gaba=config[18],
        alpha_nmda=config[19],
        v_threshold=config[20],
        v_reset=config[21],
        tau_ref=config[22],
        dt=config[23],
    )
    steps = inputs[0].size
    traces = {key: np.empty(steps, dtype=np.float64) for key in _TRACE_KEYS}
    output_events = np.empty(steps, dtype=np.int64)
    for index, (current, recurrent, external, inhibitory) in enumerate(zip(*inputs, strict=True)):
        output_events[index] = state.step(
            float(current), bool(recurrent), bool(external), bool(inhibitory)
        )
        current_state = state.get_state()
        for key, state_key in zip(
            _TRACE_KEYS,
            ("v", "s_ampa", "s_nmda", "x_nmda", "s_gaba", "ref_remaining"),
            strict=True,
        ):
            traces[key][index] = current_state[state_key]
    final_values = (
        tuple(config[:6]) if steps == 0 else tuple(float(traces[key][-1]) for key in _TRACE_KEYS)
    )
    return {
        **traces,
        "events": output_events,
        **dict(zip(_FINAL_KEYS, final_values, strict=True)),
    }


def simulate_compte_wm(
    v: float,
    s_ampa: float,
    s_nmda: float,
    x_nmda: float,
    s_gaba: float,
    ref_remaining: float,
    g_l: float,
    g_ampa: float,
    g_nmda: float,
    g_gaba: float,
    e_l: float,
    e_exc: float,
    e_inh: float,
    c_m: float,
    mg: float,
    tau_ampa: float,
    tau_nmda: float,
    tau_x: float,
    tau_gaba: float,
    alpha_nmda: float,
    v_threshold: float,
    v_reset: float,
    tau_ref: float,
    dt: float,
    currents: npt.ArrayLike,
    recurrent_events: npt.ArrayLike,
    external_events: npt.ArrayLike,
    inhibitory_events: npt.ArrayLike,
    *,
    backend: str = "auto",
) -> CompteWMResult:
    """Run the complete configured Compte contract on one real backend."""
    config = tuple(
        float(value)
        for value in (
            v,
            s_ampa,
            s_nmda,
            x_nmda,
            s_gaba,
            ref_remaining,
            g_l,
            g_ampa,
            g_nmda,
            g_gaba,
            e_l,
            e_exc,
            e_inh,
            c_m,
            mg,
            tau_ampa,
            tau_nmda,
            tau_x,
            tau_gaba,
            alpha_nmda,
            v_threshold,
            v_reset,
            tau_ref,
            dt,
        )
    )
    input_arrays = _inputs(currents, recurrent_events, external_events, inhibitory_events)
    empty_inputs = (
        input_arrays[0][:0],
        input_arrays[1][:0],
        input_arrays[2][:0],
        input_arrays[3][:0],
    )
    _python_runner(config, empty_inputs)
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown Compte backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} Compte backend is unavailable")
    if selected == "python":
        result = _python_runner(config, input_arrays)
    elif selected == "rust":
        result = _rust_runner(config, input_arrays)
    elif selected == "julia":
        result = importlib.import_module("sc_neurocore.accel.julia.neurons").simulate_compte_wm(
            *config, *input_arrays
        )
    else:
        result = cast(_NativeRunner, _native_module(selected).simulate_compte_wm)(
            *config, *input_arrays
        )
    return _normalise(result, input_arrays[0].size, config[:6])


__all__ = [
    "KERNEL",
    "PARITY_ATOL",
    "auto_backend",
    "backend_available",
    "simulate_compte_wm",
]
