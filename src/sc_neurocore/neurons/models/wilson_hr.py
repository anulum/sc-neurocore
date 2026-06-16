# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson 1999 polynomial cortical model

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional

import numpy as np
import numpy.typing as npt

# ───────────────────────── backend detection ─────────────────────────
#
# A single `step` is trivial, but an N-step RK4 simulation is a sequential
# recurrence (each step depends on the previous) that does not vectorise, so a
# compiled inner loop genuinely beats Python. The polyglot chain (Rust PyO3,
# Julia juliacall, Go cgo, Mojo FFI) accelerates `simulate`. The right-hand side
# is exact polynomial arithmetic, so Rust/Julia/Go reproduce the NumPy reference
# bit-for-bit; the FMA-fusing Mojo backend stays within a measured per-step ULP
# band with identical spike counts.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_wilson_hr_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "wilson_hr.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.WilsonHRAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "wilson_hr", "libwilsonhr.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "wilson_hr_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 5 + [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    fn.restype = ctypes.c_longlong
    _go_lib = lib
    _HAS_GO = True
    return True


def _ensure_mojo_loaded() -> bool:
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libwilsonhr.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "wilson_hr_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 5 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class WilsonHRNeuron:
    """Wilson 1999 polynomial cortical model.

    dV/dt = -(17.81 + 47.71*V + 32.63*V^2)*(V - 0.55) - 26*R*(V + 0.92) + I
    dR/dt = (-R + 1.35*V + 1.03) / tau_R

    The maintained production path advances the coupled (V, R) state with
    candidate-first RK4 and commits only finite candidates. Spike detection is
    a threshold event followed by Wilson-HR hard voltage reset.
    """

    _FINITE_FIELDS: ClassVar[tuple[str, ...]] = ("v", "r", "v_peak")
    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ("tau_r", "dt")

    v: float = -0.7
    r: float = 0.1
    tau_r: float = 1.9
    v_peak: float = 0.4
    dt: float = 0.05

    def __post_init__(self) -> None:
        for name in self._FINITE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real finite scalar")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in self._POSITIVE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real positive scalar")
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real finite scalar")
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError(f"{name} must be finite")
        return value

    def _validate_runtime_contract(self, current: float) -> float:
        current = self._finite_float("current", current)
        for name in self._FINITE_FIELDS:
            self._finite_float(name, getattr(self, name))
        for name in self._POSITIVE_FIELDS:
            value = self._finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        return current

    @staticmethod
    def _poly(v: float) -> float:
        value = -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)
        if not math.isfinite(value):
            raise FloatingPointError("Wilson-HR polynomial must be finite")
        return value

    def _derivatives(self, v: float, r: float, current: float) -> tuple[float, float]:
        if not all(math.isfinite(value) for value in (v, r, current)):
            raise FloatingPointError("Wilson-HR runtime state and current must be finite")
        poly = self._poly(v)
        syn = -26.0 * r * (v + 0.92)
        dv = poly + syn + current
        dr = (-r + 1.35 * v + 1.03) / self.tau_r
        if not math.isfinite(syn) or not math.isfinite(dv) or not math.isfinite(dr):
            raise FloatingPointError("Wilson-HR derivative must be finite")
        return dv, dr

    @staticmethod
    def _validate_candidate(v: float, r: float) -> None:
        if not math.isfinite(v) or not math.isfinite(r):
            raise FloatingPointError("Wilson-HR RK4 candidate must be finite")

    def _rk4_candidate(self, current: float) -> tuple[float, float]:
        v0, r0 = self.v, self.r
        dt = self.dt
        k1 = self._derivatives(v0, r0, current)
        k2 = self._derivatives(v0 + 0.5 * dt * k1[0], r0 + 0.5 * dt * k1[1], current)
        k3 = self._derivatives(v0 + 0.5 * dt * k2[0], r0 + 0.5 * dt * k2[1], current)
        k4 = self._derivatives(v0 + dt * k3[0], r0 + dt * k3[1], current)
        candidate = (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            r0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        )
        self._validate_candidate(*candidate)
        return candidate

    def step(self, current: float) -> int:
        current = self._validate_runtime_contract(current)
        next_v, next_r = self._rk4_candidate(current)
        self.v = next_v
        self.r = next_r
        if self.v >= self.v_peak:
            self.v = -0.7
            return 1
        return 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane variable ``v`` after step ``t`` (already
        hard-reset to ``-0.7`` on spiking steps); ``spikes`` counts the steps whose
        post-RK4 ``v`` reached ``v_peak``. The instance state ``(v, r)`` is advanced
        to the final step. The Rust/Julia/Go backends reproduce the pure-NumPy
        reference bit-for-bit; the FMA-fusing Mojo backend stays within a measured
        per-step ULP band with identical spike counts.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        current = self._validate_runtime_contract(current)

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust Wilson-HR backend requested but the engine wheel lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError(
                "Julia Wilson-HR backend requested but juliacall/.jl is unavailable."
            )
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Wilson-HR backend requested but libwilsonhr.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/wilson_hr && go build "
                "-buildmode=c-shared -o libwilsonhr.so wilson_hr.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Wilson-HR backend requested but libwilsonhr.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libwilsonhr.so wilson_hr.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, vf, rf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, vf, rf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, vf, rf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, vf, rf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, vf, rf = self._simulate_python(n_steps, current)
        self.v, self.r = vf, rf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, r = self.v, self.r
        tau_r, v_peak, dt = self.tau_r, self.v_peak, self.dt

        def deriv(vv: float, rr: float) -> tuple[float, float]:
            poly = -(17.81 + 47.71 * vv + 32.63 * vv * vv) * (vv - 0.55)
            syn = -26.0 * rr * (vv + 0.92)
            return poly + syn + current, (-rr + 1.35 * vv + 1.03) / tau_r

        spikes = 0
        for t in range(n_steps):
            dv1, dr1 = deriv(v, r)
            dv2, dr2 = deriv(v + 0.5 * dt * dv1, r + 0.5 * dt * dr1)
            dv3, dr3 = deriv(v + 0.5 * dt * dv2, r + 0.5 * dt * dr2)
            dv4, dr4 = deriv(v + dt * dv3, r + dt * dr3)
            v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
            r = r + dt * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4) / 6.0
            if v >= v_peak:
                v = -0.7
                spikes += 1
            trace[t] = v
        return trace, spikes, v, r

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, vf, rf = _rust_simulate(
            self.v, self.r, self.tau_r, self.v_peak, self.dt, n_steps, current
        )
        return np.asarray(trace_list, dtype=np.float64), int(spikes), float(vf), float(rf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.v),
            float(self.r),
            float(self.tau_r),
            float(self.v_peak),
            float(self.dt),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.vf), float(result.rf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _go_lib.wilson_hr_simulate_c(
            ctypes.c_double(self.v),
            ctypes.c_double(self.r),
            ctypes.c_double(self.tau_r),
            ctypes.c_double(self.v_peak),
            ctypes.c_double(self.dt),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        rf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, rf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _mojo_lib.wilson_hr_simulate_c(
            float(self.v),
            float(self.r),
            float(self.tau_r),
            float(self.v_peak),
            float(self.dt),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        rf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, rf

    def reset(self) -> None:
        self.v = -0.7
        self.r = 0.1
