# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — measured-order Brunel-Wang accelerator dispatch

"""Dispatch the complete four-gate midpoint-RK2 cell contract."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import with_floor
from sc_neurocore.accel.backend_selection import select_backend_order

FloatArray: TypeAlias = npt.NDArray[np.float64]
BrunelWangResult: TypeAlias = dict[str, FloatArray | npt.NDArray[np.int64] | float]
KERNEL = "brunel_wang_midpoint_rk2_batch"
PARITY_ATOL = {"python": 0.0, "rust": 2.0e-12, "julia": 2.0e-12, "go": 2.0e-12, "mojo": 2.0e-10}
_AUTO_BACKENDS = with_floor("python")


class _NativeRunner(Protocol):
    """Mapping-returning contract shared by Go and Mojo facades."""

    def __call__(self, *args: object) -> Mapping[str, object]: ...


def _engine_class() -> type[Any] | None:
    try:
        return cast(type[Any], importlib.import_module("sc_neurocore_engine").BrunelWangNeuron)
    except (ImportError, AttributeError):
        return None


def _native_module(backend: str) -> Any:
    return importlib.import_module(f"sc_neurocore.accel.{backend}.brunel_wang")


def backend_available(backend: str) -> bool:
    """Return whether one named Brunel-Wang runtime is executable."""
    if backend == "python":
        return True
    if backend == "rust":
        return _engine_class() is not None
    if backend == "julia":
        try:
            importlib.import_module("sc_neurocore.accel.julia.neurons")._ensure_brunel_wang_loaded()
        except (ImportError, FileNotFoundError, RuntimeError):
            return False
        return True
    if backend in {"go", "mojo"}:
        try:
            module = _native_module(backend)
        except ImportError:
            return False
        return bool(getattr(module, f"_HAS_{backend.upper()}_BRUNEL_WANG", False))
    return False


def auto_backend() -> str:
    """Return the first available measured lane, with Python as floor."""
    order = select_backend_order(KERNEL, static=_AUTO_BACKENDS)
    return next((backend for backend in order if backend_available(backend)), "python")


def _inputs(*arrays: npt.ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    converted = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in arrays)
    lengths = {value.size for value in converted}
    if any(value.ndim != 1 for value in converted) or len(lengths) != 1:
        raise ValueError("Brunel-Wang gate arrays must be one-dimensional and equal-length")
    if any(not np.isfinite(value).all() or np.any(value < 0.0) for value in converted):
        raise ValueError("Brunel-Wang aggregate gates must be finite and non-negative")
    return cast(tuple[FloatArray, FloatArray, FloatArray, FloatArray], converted)


def _normalise(
    result: Mapping[str, object], steps: int, initial: tuple[float, float]
) -> BrunelWangResult:
    normalised: BrunelWangResult = {}
    for key in ("voltages", "refractory"):
        values = np.ascontiguousarray(result[key], dtype=np.float64)
        if values.shape != (steps,) or not np.isfinite(values).all():
            raise FloatingPointError(f"Brunel-Wang backend returned malformed {key}")
        normalised[key] = values
    events = np.ascontiguousarray(result["events"], dtype=np.int64)
    if events.shape != (steps,) or not np.isin(events, (0, 1)).all():
        raise FloatingPointError("Brunel-Wang backend returned malformed events")
    normalised["events"] = events
    for index, key in enumerate(("v_final", "ref_final")):
        value = float(cast(float, result[key]))
        if not np.isfinite(value):
            raise FloatingPointError(f"Brunel-Wang backend returned invalid {key}")
        trace_key = "voltages" if index == 0 else "refractory"
        trace = cast(FloatArray, normalised[trace_key])
        expected = initial[index] if steps == 0 else float(trace[-1])
        if value != expected:
            raise FloatingPointError(f"Brunel-Wang {key} disagrees with its trace")
        normalised[key] = value
    if np.any(cast(FloatArray, normalised["refractory"]) < 0.0):
        raise FloatingPointError("Brunel-Wang refractory state became negative")
    return normalised


def _python_runner(
    config: tuple[float, ...], gates: tuple[FloatArray, ...]
) -> Mapping[str, object]:
    from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron

    state = BrunelWangNeuron(
        v=config[0],
        v_rest=config[2],
        v_reset=config[3],
        v_threshold=config[4],
        tau_m=config[5],
        tau_ref=config[6],
        g_ampa_ext=config[7],
        g_ampa_rec=config[8],
        g_nmda=config[9],
        g_gaba=config[10],
        v_ampa=config[11],
        v_nmda=config[12],
        v_gaba=config[13],
        C_m=config[14],
        mg_conc=config[15],
        dt=config[16],
    )
    state._ref_remaining = config[1]
    steps = gates[0].size
    voltages = np.empty(steps, dtype=np.float64)
    refractory = np.empty(steps, dtype=np.float64)
    events = np.empty(steps, dtype=np.int64)
    for index, values in enumerate(zip(*gates, strict=True)):
        events[index] = state.step(*values)
        voltages[index] = state.v
        refractory[index] = state._ref_remaining
    return {
        "voltages": voltages,
        "refractory": refractory,
        "events": events,
        "v_final": state.v,
        "ref_final": state._ref_remaining,
    }


def _rust_runner(config: tuple[float, ...], gates: tuple[FloatArray, ...]) -> Mapping[str, object]:
    engine = _engine_class()
    if engine is None:
        raise RuntimeError("Rust Brunel-Wang backend is unavailable")
    state = engine(
        v=config[0],
        v_rest=config[2],
        v_reset=config[3],
        v_threshold=config[4],
        tau_m=config[5],
        tau_ref=config[6],
        g_ampa_ext=config[7],
        g_ampa_rec=config[8],
        g_nmda=config[9],
        g_gaba=config[10],
        v_ampa=config[11],
        v_nmda=config[12],
        v_gaba=config[13],
        c_m=config[14],
        mg_conc=config[15],
        dt=config[16],
        ref_remaining=config[1],
    )
    steps = gates[0].size
    voltages = np.empty(steps, dtype=np.float64)
    refractory = np.empty(steps, dtype=np.float64)
    events = np.empty(steps, dtype=np.int64)
    for index, values in enumerate(zip(*gates, strict=True)):
        events[index] = state.step(*values)
        voltages[index], refractory[index] = state.get_state()
    v_final, ref_final = (config[0], config[1]) if steps == 0 else (voltages[-1], refractory[-1])
    return {
        "voltages": voltages,
        "refractory": refractory,
        "events": events,
        "v_final": v_final,
        "ref_final": ref_final,
    }


def simulate_brunel_wang(
    v: float,
    ref_remaining: float,
    v_rest: float,
    v_reset: float,
    v_threshold: float,
    tau_m: float,
    tau_ref: float,
    g_ampa_ext: float,
    g_ampa_rec: float,
    g_nmda: float,
    g_gaba: float,
    v_ampa: float,
    v_nmda: float,
    v_gaba: float,
    c_m: float,
    mg_conc: float,
    dt: float,
    i_ampa_ext: npt.ArrayLike,
    s_ampa_rec: npt.ArrayLike,
    s_nmda_rec: npt.ArrayLike,
    s_gaba: npt.ArrayLike,
    *,
    backend: str = "auto",
) -> BrunelWangResult:
    """Run the complete configured four-gate contract on one real backend."""
    config = tuple(
        float(value)
        for value in (
            v,
            ref_remaining,
            v_rest,
            v_reset,
            v_threshold,
            tau_m,
            tau_ref,
            g_ampa_ext,
            g_ampa_rec,
            g_nmda,
            g_gaba,
            v_ampa,
            v_nmda,
            v_gaba,
            c_m,
            mg_conc,
            dt,
        )
    )
    gates = _inputs(i_ampa_ext, s_ampa_rec, s_nmda_rec, s_gaba)
    # The golden constructor/step validator is the shared fail-closed contract.
    _python_runner(config, tuple(value[:0] for value in gates))
    selected = auto_backend() if backend == "auto" else backend
    if selected not in PARITY_ATOL:
        raise ValueError(f"unknown Brunel-Wang backend: {selected}")
    if not backend_available(selected):
        raise RuntimeError(f"{selected} Brunel-Wang backend is unavailable")
    if selected == "python":
        result = _python_runner(config, gates)
    elif selected == "rust":
        result = _rust_runner(config, gates)
    elif selected == "julia":
        result = importlib.import_module("sc_neurocore.accel.julia.neurons").simulate_brunel_wang(
            *config, *gates
        )
    else:
        result = cast(_NativeRunner, _native_module(selected).simulate_brunel_wang)(*config, *gates)
    return _normalise(result, gates[0].size, (v, ref_remaining))


__all__ = ["KERNEL", "PARITY_ATOL", "auto_backend", "backend_available", "simulate_brunel_wang"]
