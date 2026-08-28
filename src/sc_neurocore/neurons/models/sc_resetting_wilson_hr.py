# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — historical resetting Wilson-HR project recurrence

"""Explicit identity for the historical SC resetting Wilson-HR recurrence."""

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, cast

import numpy as np
import numpy.typing as npt


_RustSimulate = Callable[..., "tuple[list[float], int, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return cast(_RustSimulate, engine.py_sc_resetting_wilson_hr_simulate)


try:
    _rust_simulate: Optional[_RustSimulate] = _load_rust_simulate()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _rust_simulate = None
    _HAS_RUST = False

_julia_module = None
_HAS_JULIA = False
_go_lib = None
_HAS_GO = False
_mojo_lib = None
_HAS_MOJO = False

_ACCEL_ROOT = _os.path.join(_os.path.dirname(__file__), "..", "..", "accel")


def _ensure_julia_loaded() -> bool:
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    import importlib.util as importlib_util

    if importlib_util.find_spec("juliacall") is None:
        return False
    path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "julia", "neurons", "sc_resetting_wilson_hr.jl")
    )
    if not _os.path.isfile(path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(path)
    _julia_module = jl.SCResettingWilsonHRAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    path = _os.path.abspath(
        _os.path.join(
            _ACCEL_ROOT,
            "go",
            "neurons",
            "sc_resetting_wilson_hr",
            "libsc_resetting_wilson_hr.so",
        )
    )
    if not _os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    function = getattr(library, "sc_resetting_wilson_hr_simulate_c", None)
    if function is None:
        return False
    function.argtypes = [ctypes.c_double] * 5 + [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    function.restype = ctypes.c_longlong
    _go_lib = library
    _HAS_GO = True
    return True


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    import ctypes

    path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libsc_resetting_wilson_hr.so")
    )
    if not _os.path.isfile(path):
        return False
    try:
        library = ctypes.CDLL(path)
    except OSError:
        return False
    function = getattr(library, "sc_resetting_wilson_hr_simulate_c", None)
    if function is None:
        return False
    function.argtypes = [ctypes.c_double] * 5 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
    ]
    function.restype = ctypes.c_int64
    _mojo_lib = library
    _HAS_MOJO = True
    return True


@dataclass
class SCResettingWilsonHRNeuron:
    """Preserve the former unit-capacitance, hard-reset project recurrence.

    This model is an SC-NeuroCore specialisation. It is not attributed to the
    continuous Wilson (1999) equations, which are implemented by
    :class:`~sc_neurocore.neurons.models.wilson_hr.WilsonHRNeuron`.
    """

    _FINITE_FIELDS: ClassVar[tuple[str, ...]] = ("v", "r", "v_peak")
    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ("tau_r", "dt")

    v: float = -0.7
    r: float = 0.1
    tau_r: float = 1.9
    v_peak: float = 0.4
    dt: float = 0.05

    def __post_init__(self) -> None:
        validated: dict[str, float] = {}
        for name in self._FINITE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            validated[name] = value
        for name in self._POSITIVE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            validated[name] = value
        for name, value in validated.items():
            setattr(self, name, value)

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real scalar")
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError(f"{name} must be finite")
        return value

    def _validate_runtime_contract(self, current: float) -> float:
        current = self._finite_float("current", current)
        validated: dict[str, float] = {}
        for name in self._FINITE_FIELDS:
            validated[name] = self._finite_float(name, getattr(self, name))
        for name in self._POSITIVE_FIELDS:
            value = self._finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            validated[name] = value
        for name, value in validated.items():
            setattr(self, name, value)
        return current

    def _derivatives(self, v: float, r: float, current: float) -> tuple[float, float]:
        polynomial = -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)
        recovery_current = -26.0 * r * (v + 0.92)
        candidate = polynomial + recovery_current + current, (-r + 1.35 * v + 1.03) / self.tau_r
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("SC resetting Wilson-HR derivative became non-finite")
        return candidate

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        v, r, dt = self.v, self.r, self.dt
        k1 = self._derivatives(v, r, current)
        k2 = self._derivatives(v + 0.5 * dt * k1[0], r + 0.5 * dt * k1[1], current)
        k3 = self._derivatives(v + 0.5 * dt * k2[0], r + 0.5 * dt * k2[1], current)
        k4 = self._derivatives(v + dt * k3[0], r + dt * k3[1], current)
        candidate = (
            v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            r + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        )
        if not all(math.isfinite(value) for value in candidate):
            raise FloatingPointError("SC resetting Wilson-HR candidate became non-finite")
        return candidate

    def step(self, current: float) -> int:
        """Advance one project-RK4 step and apply the historical hard reset."""
        current = self._validate_runtime_contract(current)
        next_v, next_r = self._rk4_candidate(current)
        spiked = next_v >= self.v_peak
        self.v = -0.7 if spiked else next_v
        self.r = next_r
        return int(spiked)

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Run a failure-atomic batch through a selected maintained runtime.

        Parameters
        ----------
        n_steps:
            Number of constant-current RK4 updates.
        current:
            Finite applied current for the complete batch.
        backend:
            ``auto``, ``rust``, ``julia``, ``go``, ``mojo``, or ``python``.

        Returns
        -------
        tuple[numpy.ndarray, int]
            Post-step voltage trace and hard-reset event count.

        Raises
        ------
        TypeError
            If a numeric input has the wrong scalar type.
        ValueError
            If a configuration or backend selection is invalid.
        FloatingPointError
            If any runtime rejects a non-finite stage or batch candidate. The
            instance state remains unchanged for rejected batch arithmetic.
        """
        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise TypeError("n_steps must be an integer")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        current = self._validate_runtime_contract(current)
        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust SC resetting Wilson-HR backend unavailable; build engine")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia SC resetting Wilson-HR backend unavailable")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go SC resetting Wilson-HR backend unavailable; build "
                "accel/go/neurons/sc_resetting_wilson_hr/libsc_resetting_wilson_hr.so"
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo SC resetting Wilson-HR backend unavailable; build "
                "accel/mojo/neurons/libsc_resetting_wilson_hr.so"
            )
        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, events, final_v, final_r = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, events, final_v, final_r = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, events, final_v, final_r = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, events, final_v, final_r = self._simulate_mojo(n_steps, current)
        else:
            trace, events, final_v, final_r = self._simulate_python(n_steps, current)
        trace, events, final_v, final_r = self._validate_batch_result(
            backend=backend,
            n_steps=n_steps,
            trace=trace,
            events=events,
            final_state=(final_v, final_r),
        )
        self.v, self.r = final_v, final_r
        return trace, events

    @staticmethod
    def _validate_batch_result(
        *,
        backend: str,
        n_steps: int,
        trace: npt.NDArray[np.float64],
        events: int,
        final_state: tuple[float, float],
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        values = np.asarray(trace, dtype=np.float64)
        if (
            events < 0
            or events > n_steps
            or values.shape != (n_steps,)
            or not np.all(np.isfinite(values))
            or not all(math.isfinite(value) for value in final_state)
        ):
            raise FloatingPointError(
                f"SC resetting Wilson-HR {backend} batch produced an invalid candidate"
            )
        return values, int(events), final_state[0], final_state[1]

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, r = self.v, self.r
        tau_r, v_peak, dt = self.tau_r, self.v_peak, self.dt

        def derivatives(voltage: float, recovery: float) -> tuple[float, float]:
            polynomial = -(17.81 + 47.71 * voltage + 32.63 * voltage * voltage) * (voltage - 0.55)
            recovery_current = -26.0 * recovery * (voltage + 0.92)
            candidate = (
                polynomial + recovery_current + current,
                (-recovery + 1.35 * voltage + 1.03) / tau_r,
            )
            if not all(math.isfinite(value) for value in candidate):
                raise FloatingPointError(
                    "SC resetting Wilson-HR Python batch derivative became non-finite"
                )
            return candidate

        events = 0
        for index in range(n_steps):
            dv1, dr1 = derivatives(v, r)
            dv2, dr2 = derivatives(v + 0.5 * dt * dv1, r + 0.5 * dt * dr1)
            dv3, dr3 = derivatives(v + 0.5 * dt * dv2, r + 0.5 * dt * dr2)
            dv4, dr4 = derivatives(v + dt * dv3, r + dt * dr3)
            candidate = (
                v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0,
                r + dt * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4) / 6.0,
            )
            if not all(math.isfinite(value) for value in candidate):
                raise FloatingPointError(
                    "SC resetting Wilson-HR Python batch candidate became non-finite"
                )
            v, r = candidate
            if v >= v_peak:
                v = -0.7
                events += 1
            trace[index] = v
        return trace, events, v, r

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace, events, final_v, final_r = _rust_simulate(
            self.v,
            self.r,
            self.tau_r,
            self.v_peak,
            self.dt,
            n_steps,
            current,
        )
        return (
            np.asarray(trace, dtype=np.float64),
            int(events),
            float(final_v),
            float(final_r),
        )

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        try:
            result = _julia_module.simulate_trace(
                float(self.v),
                float(self.r),
                float(self.tau_r),
                float(self.v_peak),
                float(self.dt),
                int(n_steps),
                float(current),
            )
        except Exception as exc:
            raise FloatingPointError(
                "SC resetting Wilson-HR Julia batch rejected an invalid candidate"
            ) from exc
        return (
            np.asarray(result.trace, dtype=np.float64),
            int(result.spikes),
            float(result.vf),
            float(result.rf),
        )

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _go_lib.sc_resetting_wilson_hr_simulate_c(
            ctypes.c_double(self.v),
            ctypes.c_double(self.r),
            ctypes.c_double(self.tau_r),
            ctypes.c_double(self.v_peak),
            ctypes.c_double(self.dt),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            float(trace[n_steps]),
            float(trace[n_steps + 1]),
        )

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        events = _mojo_lib.sc_resetting_wilson_hr_simulate_c(
            float(self.v),
            float(self.r),
            float(self.tau_r),
            float(self.v_peak),
            float(self.dt),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        return (
            np.ascontiguousarray(trace[:n_steps]),
            int(events),
            float(trace[n_steps]),
            float(trace[n_steps + 1]),
        )

    def reset(self) -> None:
        """Restore the historical project state."""
        self.v = -0.7
        self.r = 0.1
