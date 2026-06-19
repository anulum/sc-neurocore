# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Allen Institute GLIF5 candidate-first RK4 dynamics

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

# ───────────────────────── backend detection ─────────────────────────
#
# A single `step` is trivial, but an N-step RK4 simulation is a sequential
# recurrence (each step depends on the previous, plus a discontinuous spike
# reset) that does not vectorise, so a compiled inner loop genuinely beats
# Python. The polyglot chain (Rust PyO3, Julia juliacall, Go cgo, Mojo FFI)
# accelerates `simulate`. The Allen GLIF5 right-hand side is purely linear —
# additions, multiplications and divisions, no transcendental functions — so
# every RK4 stage is exact arithmetic and Rust, Julia and Go reproduce the
# NumPy reference bit-for-bit. Mojo fuses multiply-add (FMA), so it is validated
# per-step and on spike counts rather than bit-exactly.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_glif_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "glif.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.GlifAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "go", "neurons", "glif", "libglif.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "glif_simulate_c", None)
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libglif.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "glif_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 17 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class GLIFNeuron:
    """Allen Institute GLIF5 generalised leaky integrate-and-fire neuron.

    The four dynamic states are advanced with candidate-first RK4 over the
    continuous GLIF flow. Spike reset is applied only after the candidate is
    finite and crosses the adaptive threshold.

    Reference: Teeter, C. et al. (2018). Nat. Commun. 9:709.
    """

    v: float = -70.0
    theta: float = -50.0
    theta_inf: float = -50.0
    i_asc1: float = 0.0
    i_asc2: float = 0.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    tau_m: float = 10.0
    tau_theta: float = 100.0
    tau_asc1: float = 10.0
    tau_asc2: float = 200.0
    a_theta: float = 0.01
    delta_theta: float = 2.0
    r_asc1: float = 1.0
    r_asc2: float = 0.5
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        self._raise_if_invalid_runtime()

    @staticmethod
    def _finite_values(values: tuple[float, ...]) -> bool:
        return all(math.isfinite(value) for value in values)

    def _raise_if_invalid_runtime(self) -> None:
        finite_fields = (
            "v",
            "theta",
            "theta_inf",
            "i_asc1",
            "i_asc2",
            "v_rest",
            "v_reset",
            "a_theta",
            "delta_theta",
            "r_asc1",
            "r_asc2",
            "resistance",
        )
        for field in finite_fields:
            value = getattr(self, field)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        for field in ("tau_m", "tau_theta", "tau_asc1", "tau_asc2", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        for field in ("delta_theta", "resistance"):
            value = getattr(self, field)
            if value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")

    def _derivatives(
        self,
        v: float,
        theta: float,
        i_asc1: float,
        i_asc2: float,
        current: float,
    ) -> tuple[float, float, float, float]:
        return (
            (-(v - self.v_rest) + self.resistance * current + i_asc1 + i_asc2) / self.tau_m,
            (self.theta_inf - theta + self.a_theta * (v - self.v_rest)) / self.tau_theta,
            -i_asc1 / self.tau_asc1,
            -i_asc2 / self.tau_asc2,
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
        state = (self.v, self.theta, self.i_asc1, self.i_asc2)
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
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._raise_if_invalid_runtime()
        candidate = self._rk4_candidate(current)
        if not self._finite_values(candidate):
            raise FloatingPointError("GLIF candidate state must be finite")

        next_v, next_theta, next_i_asc1, next_i_asc2 = candidate
        self.v = next_v
        self.theta = next_theta
        self.i_asc1 = next_i_asc1
        self.i_asc2 = next_i_asc2

        if self.v >= self.theta:
            self.v = self.v_reset
            self.theta += self.delta_theta
            self.i_asc1 += self.r_asc1
            self.i_asc2 += self.r_asc2
            return 1
        return 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane voltage ``v`` after step ``t`` (post-reset
        when that step fired); ``spikes`` counts the threshold crossings. The
        instance state ``(v, theta, i_asc1, i_asc2)`` is advanced to the final
        step. The Allen GLIF5 right-hand side is purely linear, so the Rust, Julia
        and Go backends reproduce the pure-NumPy reference bit-for-bit; Mojo fuses
        multiply-add and is validated per-step and on spike counts.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._raise_if_invalid_runtime()

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust GLIF backend requested but the engine wheel lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia GLIF backend requested but juliacall/.jl is unavailable.")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go GLIF backend requested but libglif.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/glif && go build "
                "-buildmode=c-shared -o libglif.so glif.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo GLIF backend requested but libglif.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libglif.so glif.mojo`."
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
        self.v, self.theta, self.i_asc1, self.i_asc2 = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, theta, i_asc1, i_asc2 = self.v, self.theta, self.i_asc1, self.i_asc2
        v_rest, v_reset = self.v_rest, self.v_reset
        theta_inf = self.theta_inf
        tau_m, tau_theta = self.tau_m, self.tau_theta
        tau_asc1, tau_asc2 = self.tau_asc1, self.tau_asc2
        a_theta, delta_theta = self.a_theta, self.delta_theta
        r_asc1, r_asc2, resistance, dt = self.r_asc1, self.r_asc2, self.resistance, self.dt
        half_dt = 0.5 * dt

        def deriv(vv: float, th: float, a1: float, a2: float) -> tuple[float, float, float, float]:
            return (
                (-(vv - v_rest) + resistance * current + a1 + a2) / tau_m,
                (theta_inf - th + a_theta * (vv - v_rest)) / tau_theta,
                -a1 / tau_asc1,
                -a2 / tau_asc2,
            )

        spikes = 0
        for t in range(n_steps):
            k1 = deriv(v, theta, i_asc1, i_asc2)
            k2 = deriv(
                v + half_dt * k1[0],
                theta + half_dt * k1[1],
                i_asc1 + half_dt * k1[2],
                i_asc2 + half_dt * k1[3],
            )
            k3 = deriv(
                v + half_dt * k2[0],
                theta + half_dt * k2[1],
                i_asc1 + half_dt * k2[2],
                i_asc2 + half_dt * k2[3],
            )
            k4 = deriv(
                v + dt * k3[0],
                theta + dt * k3[1],
                i_asc1 + dt * k3[2],
                i_asc2 + dt * k3[3],
            )
            v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
            theta = theta + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
            i_asc1 = i_asc1 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
            i_asc2 = i_asc2 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
            if v >= theta:
                v = v_reset
                theta += delta_theta
                i_asc1 += r_asc1
                i_asc2 += r_asc2
                spikes += 1
            trace[t] = v
        return trace, spikes, (v, theta, i_asc1, i_asc2)

    def _scalar_args(self, n_steps: int, current: float) -> tuple[float, ...]:
        return (
            self.v,
            self.theta,
            self.theta_inf,
            self.i_asc1,
            self.i_asc2,
            self.v_rest,
            self.v_reset,
            self.tau_m,
            self.tau_theta,
            self.tau_asc1,
            self.tau_asc2,
            self.a_theta,
            self.delta_theta,
            self.r_asc1,
            self.r_asc2,
            self.resistance,
            self.dt,
        )

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        assert _rust_simulate is not None
        trace_list, spikes, vf, theta_f, a1_f, a2_f = _rust_simulate(
            *self._scalar_args(n_steps, current), n_steps, current
        )
        return (
            np.asarray(trace_list, dtype=np.float64),
            int(spikes),
            (float(vf), float(theta_f), float(a1_f), float(a2_f)),
        )

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            *(float(x) for x in self._scalar_args(n_steps, current)),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return (
            trace,
            int(result.spikes),
            (float(result.vf), float(result.theta_f), float(result.a1_f), float(result.a2_f)),
        )

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float, float, float]]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 4, dtype=np.float64, order="C")
        spikes = _go_lib.glif_simulate_c(
            *(ctypes.c_double(x) for x in self._scalar_args(n_steps, current)),
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
        spikes = _mojo_lib.glif_simulate_c(
            *(float(x) for x in self._scalar_args(n_steps, current)),
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
            return self.v, self.theta, self.i_asc1, self.i_asc2
        return (
            float(trace[n_steps]),
            float(trace[n_steps + 1]),
            float(trace[n_steps + 2]),
            float(trace[n_steps + 3]),
        )

    def reset(self) -> None:
        self.v = self.v_rest
        self.theta = self.theta_inf
        self.i_asc1 = 0.0
        self.i_asc2 = 0.0
