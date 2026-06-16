# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mihalas-Niebur Generalized IF candidate-first RK4 dynamics

from __future__ import annotations

import importlib as _importlib
import os as _os
from dataclasses import dataclass
from math import isfinite
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

# ───────────────────────── backend detection ─────────────────────────
#
# A single `step` is trivial, but an N-step RK4 simulation is a sequential
# recurrence (each step depends on the previous, plus a discontinuous spike
# reset) that does not vectorise, so a compiled inner loop genuinely beats
# Python. The polyglot chain (Rust PyO3, Julia juliacall, Go cgo, Mojo FFI)
# accelerates `simulate`. The Mihalas-Niebur 2009 right-hand side is purely
# linear — additions, multiplications and divisions, no transcendental
# functions — so every RK4 stage is exact arithmetic and Rust, Julia and Go
# reproduce the NumPy reference bit-for-bit. Mojo fuses multiply-add (FMA), so
# it is validated per-step and on spike counts rather than bit-exactly.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_mihalas_niebur_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "mihalas_niebur.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.MihalasNieburAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "mihalas_niebur", "libmihalasniebur.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "mihalas_niebur_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 17 + [
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libmihalasniebur.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "mihalas_niebur_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 17 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class MihalasNieburNeuron:
    """Mihalas-Niebur generalised integrate-and-fire neuron.

    The continuous four-state flow is advanced with a candidate-first RK4
    integrator. Spike reset is applied only after the continuous candidate is
    finite and crosses the adaptive threshold.

    Reference: Mihalas, S. & Niebur, E. (2009). Neural Comput. 21:704-718.
    """

    v: float = 0.0
    theta: float = 1.0
    i1: float = 0.0
    i2: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    theta_reset: float = 1.0
    theta_inf: float = 1.0
    tau_v: float = 10.0
    tau_theta: float = 100.0
    tau_1: float = 10.0
    tau_2: float = 200.0
    a: float = 0.0
    b: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    dt: float = 1.0

    @staticmethod
    def _finite_values(values: tuple[float, ...]) -> bool:
        return all(isfinite(value) for value in values)

    def _valid_runtime(self) -> bool:
        values = (
            self.v,
            self.theta,
            self.i1,
            self.i2,
            self.v_rest,
            self.v_reset,
            self.theta_reset,
            self.theta_inf,
            self.tau_v,
            self.tau_theta,
            self.tau_1,
            self.tau_2,
            self.a,
            self.b,
            self.r1,
            self.r2,
            self.dt,
        )
        return (
            self._finite_values(values)
            and self.tau_v > 0.0
            and self.tau_theta > 0.0
            and self.tau_1 > 0.0
            and self.tau_2 > 0.0
            and self.dt > 0.0
        )

    def _derivatives(
        self,
        v: float,
        theta: float,
        i1: float,
        i2: float,
        current: float,
    ) -> tuple[float, float, float, float]:
        return (
            (-(v - self.v_rest) + i1 + i2 + current) / self.tau_v,
            (self.theta_inf - theta + self.a * (v - self.v_rest)) / self.tau_theta,
            -i1 / self.tau_1,
            -i2 / self.tau_2,
        )

    @staticmethod
    def _add_scaled(
        state: tuple[float, float, float, float],
        slope: tuple[float, float, float, float],
        scale: float,
    ) -> tuple[float, float, float, float]:
        return (
            state[0] + scale * slope[0],
            state[1] + scale * slope[1],
            state[2] + scale * slope[2],
            state[3] + scale * slope[3],
        )

    def _rk4_candidate(self, current: float) -> tuple[float, float, float, float]:
        state = (self.v, self.theta, self.i1, self.i2)
        half_dt = 0.5 * self.dt
        k1 = self._derivatives(*state, current)
        k2 = self._derivatives(*self._add_scaled(state, k1, half_dt), current)
        k3 = self._derivatives(*self._add_scaled(state, k2, half_dt), current)
        k4 = self._derivatives(*self._add_scaled(state, k3, self.dt), current)
        return (
            state[0] + self.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + self.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + self.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
            state[3] + self.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        )

    def step(self, current: float) -> int:
        if not isfinite(current) or not self._valid_runtime():
            return 0

        candidate = self._rk4_candidate(current)
        if not self._finite_values(candidate):
            return 0

        next_v, next_theta, next_i1, next_i2 = candidate
        self.v = next_v
        self.theta = next_theta
        self.i1 = next_i1
        self.i2 = next_i2

        if self.v >= self.theta:
            self.v = self.v_reset + self.b * (self.v - self.v_rest)
            self.theta = max(self.theta, self.theta_reset)
            self.i1 += self.r1
            self.i2 += self.r2
            return 1
        return 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane voltage ``v`` after step ``t`` (post-reset
        when that step fired); ``spikes`` counts the threshold crossings. The
        instance state ``(v, theta, i1, i2)`` is advanced to the final step. The
        Mihalas-Niebur right-hand side is purely linear, so the Rust, Julia and Go
        backends reproduce the pure-NumPy reference bit-for-bit; Mojo fuses
        multiply-add and is validated per-step and on spike counts.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        if not isfinite(current):
            raise ValueError("current must be finite")
        if not self._valid_runtime():
            raise ValueError("Mihalas-Niebur runtime parameters must be finite and positive")

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError(
                "Rust Mihalas-Niebur backend requested but the engine wheel lacks it."
            )
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError(
                "Julia Mihalas-Niebur backend requested but juliacall/.jl is unavailable."
            )
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Mihalas-Niebur backend requested but libmihalasniebur.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/mihalas_niebur && go build "
                "-buildmode=c-shared -o libmihalasniebur.so mihalas_niebur.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Mihalas-Niebur backend requested but libmihalasniebur.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libmihalasniebur.so mihalas_niebur.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, state = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, state = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, state = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, state = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.v, self.theta, self.i1, self.i2 = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, theta, i1, i2 = self.v, self.theta, self.i1, self.i2
        v_rest, v_reset = self.v_rest, self.v_reset
        theta_reset, theta_inf = self.theta_reset, self.theta_inf
        tau_v, tau_theta, tau_1, tau_2 = self.tau_v, self.tau_theta, self.tau_1, self.tau_2
        a, b, r1, r2, dt = self.a, self.b, self.r1, self.r2, self.dt
        half_dt = 0.5 * dt

        def deriv(vv: float, th: float, j1: float, j2: float) -> tuple[float, float, float, float]:
            return (
                (-(vv - v_rest) + j1 + j2 + current) / tau_v,
                (theta_inf - th + a * (vv - v_rest)) / tau_theta,
                -j1 / tau_1,
                -j2 / tau_2,
            )

        spikes = 0
        for t in range(n_steps):
            k1 = deriv(v, theta, i1, i2)
            k2 = deriv(
                v + half_dt * k1[0],
                theta + half_dt * k1[1],
                i1 + half_dt * k1[2],
                i2 + half_dt * k1[3],
            )
            k3 = deriv(
                v + half_dt * k2[0],
                theta + half_dt * k2[1],
                i1 + half_dt * k2[2],
                i2 + half_dt * k2[3],
            )
            k4 = deriv(
                v + dt * k3[0],
                theta + dt * k3[1],
                i1 + dt * k3[2],
                i2 + dt * k3[3],
            )
            v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
            theta = theta + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
            i1 = i1 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
            i2 = i2 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
            if v >= theta:
                v = v_reset + b * (v - v_rest)
                theta = max(theta, theta_reset)
                i1 += r1
                i2 += r2
                spikes += 1
            trace[t] = v
        return trace, spikes, (v, theta, i1, i2)

    def _rust_args(self, n_steps: int, current: float) -> tuple:
        return (
            self.v,
            self.theta,
            self.i1,
            self.i2,
            self.v_rest,
            self.v_reset,
            self.theta_reset,
            self.theta_inf,
            self.tau_v,
            self.tau_theta,
            self.tau_1,
            self.tau_2,
            self.a,
            self.b,
            self.r1,
            self.r2,
            self.dt,
            n_steps,
            current,
        )

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        assert _rust_simulate is not None
        trace_list, spikes, vf, theta_f, i1_f, i2_f = _rust_simulate(
            *self._rust_args(n_steps, current)
        )
        return (
            np.asarray(trace_list, dtype=np.float64),
            int(spikes),
            (float(vf), float(theta_f), float(i1_f), float(i2_f)),
        )

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            *(float(x) for x in self._rust_args(n_steps, current)[:17]),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return (
            trace,
            int(result.spikes),
            (float(result.vf), float(result.theta_f), float(result.i1_f), float(result.i2_f)),
        )

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 4, dtype=np.float64, order="C")
        scalars = self._rust_args(n_steps, current)[:17]
        spikes = _go_lib.mihalas_niebur_simulate_c(
            *(ctypes.c_double(x) for x in scalars),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        state = self._final_state(trace, n_steps)
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), state

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 4, dtype=np.float64, order="C")
        scalars = self._rust_args(n_steps, current)[:17]
        spikes = _mojo_lib.mihalas_niebur_simulate_c(
            *(float(x) for x in scalars),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        state = self._final_state(trace, n_steps)
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), state

    def _final_state(
        self, trace: npt.NDArray[np.float64], n_steps: int
    ) -> tuple[float, float, float, float]:
        if n_steps == 0:
            return self.v, self.theta, self.i1, self.i2
        return (
            float(trace[n_steps]),
            float(trace[n_steps + 1]),
            float(trace[n_steps + 2]),
            float(trace[n_steps + 3]),
        )

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta = self.theta_reset
        self.i1 = 0.0
        self.i2 = 0.0
