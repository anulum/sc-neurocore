# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich 2007 biophysical spiking neuron

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np
import numpy.typing as npt

from ..base import BaseNeuron

# ───────────────────────── backend detection ─────────────────────────
#
# `step` runs one RK4 update; an N-step `simulate` is a sequential recurrence
# (each step depends on the previous) that does not vectorise, so a compiled
# inner loop genuinely beats Python. The right-hand side `k (v-vr)(v-vt)/C` is
# exact arithmetic — products, additions and a division, no transcendental
# functions — so Rust, Julia and Go reproduce the NumPy trace bit-for-bit; Mojo
# agrees to a small, non-amplifying ULP bound.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_izhikevich2007_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "izhikevich2007.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.Izhikevich2007Accel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "izhikevich2007", "libizh2007.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "izhikevich2007_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 12 + [
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libizh2007.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "izhikevich2007_simulate_c", None)
    if fn is None:
        return False
    # v0, u0, C, k, vr, vt, vpeak, a, b, c, d, dt (12 doubles) + n_steps + current + addr.
    fn.argtypes = [ctypes.c_double] * 12 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class Izhikevich2007Neuron(BaseNeuron):
    """Izhikevich 2007 biophysical quadratic integrate-and-fire neuron.

    Equations from Izhikevich, E. M. (2007), *Dynamical Systems in
    Neuroscience*, using the NeuroML 2 ``izhikevich2007Cell`` parameterisation:

    ``C dv/dt = k (v - vr) (v - vt) - u + I``
    ``du/dt = a (b (v - vr) - u)``

    If ``v >= vpeak`` after integration, the neuron emits one spike and applies
    ``v <- c`` and ``u <- u + d``. Units are the NeuroML base units used by the
    importer: pF, nS, mV, ms, and pA.
    """

    C: float = 100.0
    k: float = 0.7
    vr: float = -60.0
    vt: float = -40.0
    vpeak: float = 35.0
    a: float = 0.03
    b: float = -2.0
    c: float = -50.0
    d: float = 100.0
    v0: float | None = None
    dt: float = 0.1
    integrator: Literal["euler", "rk4"] = "rk4"
    v: float = field(init=False, default=0.0)
    u: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.integrator not in {"euler", "rk4"}:
            raise ValueError(f"Unsupported integrator for Izhikevich2007Neuron: {self.integrator}")
        for name in ("k", "vr", "vt", "vpeak", "a", "b", "c", "d"):
            self._require_finite(name, getattr(self, name))
        self.C = self._require_positive("C", self.C)
        self.dt = self._require_positive("dt", self.dt)
        if self.v0 is None:
            self.v0 = self.vr
        else:
            self.v0 = self._require_finite("v0", self.v0)
        self.reset_state()

    @staticmethod
    def _require_finite(name: str, value: float) -> float:
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return float(value)

    @classmethod
    def _require_positive(cls, name: str, value: float) -> float:
        result = cls._require_finite(name, value)
        if result <= 0.0:
            raise ValueError(f"{name} must be positive")
        return result

    def _rhs(self, v: float, u: float, input_current: float) -> tuple[float, float]:
        dv = (self.k * (v - self.vr) * (v - self.vt) - u + input_current) / self.C
        du = self.a * (self.b * (v - self.vr) - u)
        return dv, du

    def step(self, input_current: float) -> int:
        input_current = self._require_finite("input_current", input_current)
        if self.integrator == "euler":
            self._step_euler(input_current)
        else:
            self._step_rk4(input_current)
        return self._apply_threshold_reset()

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 steps from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane potential ``v`` after step ``t`` (already
        reset to ``c`` on a spiking step); ``spikes`` counts the steps that
        reached ``vpeak``. The instance state ``(v, u)`` is advanced to the final
        step. The compiled backends implement the **RK4** integrator (the
        production default), so this requires ``integrator == "rk4"``. The
        right-hand side is exact arithmetic, so Rust, Julia and Go reproduce the
        pure-NumPy reference bit-for-bit; Mojo agrees to a non-amplifying ULP
        bound.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        current = self._require_finite("current", current)
        if self.integrator != "rk4":
            raise ValueError(
                "simulate() accelerates the RK4 integrator only; this neuron uses "
                f"integrator={self.integrator!r}. Call step() in a loop, or set integrator='rk4'."
            )
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust Izhikevich2007 backend requested but the engine lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia Izhikevich2007 backend requested but it is unavailable.")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Izhikevich2007 backend requested but libizh2007.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/izhikevich2007 && go build "
                "-buildmode=c-shared -o libizh2007.so izhikevich2007.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Izhikevich2007 backend requested but libizh2007.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libizh2007.so izhikevich2007.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, vf, uf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, vf, uf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, vf, uf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, vf, uf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, vf, uf = self._simulate_python(n_steps, current)
        self.v, self.u = vf, uf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, u = self.v, self.u
        cap, k, vr, vt, vpeak = self.C, self.k, self.vr, self.vt, self.vpeak
        a, b, c, d, dt = self.a, self.b, self.c, self.d, self.dt
        dt6 = dt / 6.0
        spikes = 0

        def rhs(vv: float, uu: float) -> tuple[float, float]:
            return (
                (k * (vv - vr) * (vv - vt) - uu + current) / cap,
                a * (b * (vv - vr) - uu),
            )

        for t in range(n_steps):
            k1v, k1u = rhs(v, u)
            k2v, k2u = rhs(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u)
            k3v, k3u = rhs(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u)
            k4v, k4u = rhs(v + dt * k3v, u + dt * k3u)
            v = v + dt6 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            u = u + dt6 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            if v >= vpeak:
                v = c
                u = u + d
                spikes += 1
            trace[t] = v
        return trace, spikes, v, u

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, vf, uf = _rust_simulate(
            self.v,
            self.u,
            self.C,
            self.k,
            self.vr,
            self.vt,
            self.vpeak,
            self.a,
            self.b,
            self.c,
            self.d,
            self.dt,
            n_steps,
            current,
        )
        return np.asarray(trace_list, dtype=np.float64), int(spikes), float(vf), float(uf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.v),
            float(self.u),
            float(self.C),
            float(self.k),
            float(self.vr),
            float(self.vt),
            float(self.vpeak),
            float(self.a),
            float(self.b),
            float(self.c),
            float(self.d),
            float(self.dt),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.vf), float(result.uf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _go_lib.izhikevich2007_simulate_c(
            ctypes.c_double(self.v),
            ctypes.c_double(self.u),
            ctypes.c_double(self.C),
            ctypes.c_double(self.k),
            ctypes.c_double(self.vr),
            ctypes.c_double(self.vt),
            ctypes.c_double(self.vpeak),
            ctypes.c_double(self.a),
            ctypes.c_double(self.b),
            ctypes.c_double(self.c),
            ctypes.c_double(self.d),
            ctypes.c_double(self.dt),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        uf = float(trace[n_steps + 1]) if n_steps > 0 else self.u
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, uf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _mojo_lib.izhikevich2007_simulate_c(
            float(self.v),
            float(self.u),
            float(self.C),
            float(self.k),
            float(self.vr),
            float(self.vt),
            float(self.vpeak),
            float(self.a),
            float(self.b),
            float(self.c),
            float(self.d),
            float(self.dt),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        uf = float(trace[n_steps + 1]) if n_steps > 0 else self.u
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, uf

    def _step_euler(self, input_current: float) -> None:
        dv, du = self._rhs(self.v, self.u, input_current)
        self.v += self.dt * dv
        self.u += self.dt * du

    def _step_rk4(self, input_current: float) -> None:
        state = np.array([self.v, self.u], dtype=np.float64)

        def rhs(state_vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            dv, du = self._rhs(float(state_vec[0]), float(state_vec[1]), input_current)
            return np.array([dv, du], dtype=np.float64)

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * self.dt * k1)
        k3 = rhs(state + 0.5 * self.dt * k2)
        k4 = rhs(state + self.dt * k3)
        state = state + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.v = float(state[0])
        self.u = float(state[1])

    def _apply_threshold_reset(self) -> int:
        if self.v >= self.vpeak:
            self.v = self.c
            self.u += self.d
            return 1
        return 0

    def reset_state(self) -> None:
        v0 = self.vr if self.v0 is None else self.v0
        self.v = float(v0)
        self.u = self.b * (self.v - self.vr)

    def get_state(self) -> dict[str, Any]:
        return {"v": float(self.v), "u": float(self.u)}
