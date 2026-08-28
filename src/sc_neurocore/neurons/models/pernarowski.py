# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pernarowski 1994 pancreatic beta-cell burster

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
# is exact polynomial arithmetic (the cubic uses `v*v*v`, matching the engine's
# `v.powi(3)`), so Rust/Julia/Go reproduce the NumPy reference bit-for-bit; the
# FMA-fusing Mojo backend stays within a measured per-step ULP band with
# identical spike counts.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_pernarowski_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "pernarowski.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.PernarowskiAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "pernarowski", "libpernarowski.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "pernarowski_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 10 + [
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libpernarowski.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "pernarowski_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 10 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class PernarowskiNeuron:
    """Pernarowski 1994 pancreatic beta-cell burster.

    Three coupled ODEs over ``(v, w, z)`` with one fast cubic state and
    two slower recovery/adaptation variables. The public implementation uses
    candidate-first RK4 integration and preserves continuous threshold-crossing
    semantics without an artificial reset during normal evolution.

    Reference: Pernarowski, M. (1994). SIAM J. Appl. Math. 54:814–832.
    """

    _FINITE_FIELDS: ClassVar[tuple[str, ...]] = (
        "v",
        "w",
        "z",
        "alpha",
        "beta",
        "v_threshold",
    )
    _POSITIVE_FIELDS: ClassVar[tuple[str, ...]] = ("eps1", "eps2", "gamma", "dt")

    v: float = -1.0
    w: float = 0.0
    z: float = 0.0
    alpha: float = 0.1
    beta: float = 0.5
    eps1: float = 0.1
    eps2: float = 0.001
    gamma: float = 0.5
    dt: float = 0.1
    v_threshold: float = 0.5

    def __post_init__(self) -> None:
        validated: dict[str, float] = {}
        for name in self._FINITE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real finite scalar")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            validated[name] = float(value)
        for name in self._POSITIVE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real positive scalar")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            validated[name] = float(value)
        for name, value in validated.items():
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

    def _derivatives(
        self, v: float, w: float, z: float, current: float
    ) -> tuple[float, float, float]:
        if not all(math.isfinite(value) for value in (v, w, z, current)):
            raise FloatingPointError("Pernarowski runtime state and current must be finite")
        # Use v*v*v (not v**3) so the cubic is bit-identical to the engine's
        # `v.powi(3)` across the polyglot chain. Float multiplication overflows to
        # `inf` instead of raising OverflowError, so the finite guard below catches
        # the non-finite case with the same no-mutation contract.
        dv = v - v * v * v / 3.0 - w - z + current
        dw = self.eps1 * (v - self.gamma * w + self.alpha)
        dz = self.eps2 * (self.beta * (v + 0.7) - z)
        if not all(math.isfinite(value) for value in (dv, dw, dz)):
            raise FloatingPointError("Pernarowski derivative must be finite")
        return dv, dw, dz

    @staticmethod
    def _validate_candidate(v: float, w: float, z: float) -> None:
        if not all(math.isfinite(value) for value in (v, w, z)):
            raise FloatingPointError("Pernarowski RK4 candidate must be finite")

    def _rk4_candidate(self, current: float) -> tuple[float, float, float]:
        v0, w0, z0 = self.v, self.w, self.z
        dt = self.dt
        k1 = self._derivatives(v0, w0, z0, current)
        k2 = self._derivatives(
            v0 + 0.5 * dt * k1[0],
            w0 + 0.5 * dt * k1[1],
            z0 + 0.5 * dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            v0 + 0.5 * dt * k2[0],
            w0 + 0.5 * dt * k2[1],
            z0 + 0.5 * dt * k2[2],
            current,
        )
        k4 = self._derivatives(v0 + dt * k3[0], w0 + dt * k3[1], z0 + dt * k3[2], current)
        candidate = (
            v0 + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            w0 + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            z0 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        )
        self._validate_candidate(*candidate)
        return candidate

    def step(self, current: float = 0.0) -> int:
        current = self._validate_runtime_contract(current)
        v_prev = self.v
        v_new, w_new, z_new = self._rk4_candidate(current)
        self.v, self.w, self.z = v_new, w_new, z_new
        if self.v >= self.v_threshold and v_prev < self.v_threshold:
            return 1
        return 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 updates from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the fast variable ``v`` after step ``t``; ``spikes`` counts
        the steps whose ``v`` crossed ``v_threshold`` upward. The instance state
        ``(v, w, z)`` is advanced to the final step. The Rust/Julia/Go backends
        reproduce the pure-NumPy reference bit-for-bit; the FMA-fusing Mojo backend
        stays within a measured per-step ULP band with identical spike counts.
        A rejected derivative, candidate, or native result leaves the complete
        pre-batch state unchanged.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")
        current = self._validate_runtime_contract(current)

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust Pernarowski backend requested but the engine wheel lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError(
                "Julia Pernarowski backend requested but juliacall/.jl is unavailable."
            )
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Pernarowski backend requested but libpernarowski.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/pernarowski && go build "
                "-buildmode=c-shared -o libpernarowski.so pernarowski.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Pernarowski backend requested but libpernarowski.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libpernarowski.so pernarowski.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, vf, wf, zf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, vf, wf, zf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, vf, wf, zf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, vf, wf, zf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, vf, wf, zf = self._simulate_python(n_steps, current)
        trace, spikes, vf, wf, zf = self._validate_batch_result(
            backend=backend,
            n_steps=n_steps,
            trace=trace,
            spikes=spikes,
            final_state=(vf, wf, zf),
        )
        self.v, self.w, self.z = vf, wf, zf
        return trace, spikes

    @staticmethod
    def _validate_batch_result(
        *,
        backend: str,
        n_steps: int,
        trace: npt.NDArray[np.float64],
        spikes: int,
        final_state: tuple[float, float, float],
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        """Validate a complete backend result before committing instance state."""
        values = np.asarray(trace, dtype=np.float64)
        if (
            spikes < 0
            or values.shape != (n_steps,)
            or not np.all(np.isfinite(values))
            or not all(math.isfinite(value) for value in final_state)
        ):
            raise FloatingPointError(f"Pernarowski {backend} batch produced an invalid candidate")
        return values, int(spikes), final_state[0], final_state[1], final_state[2]

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        v, w, z = self.v, self.w, self.z
        alpha, beta, eps1, eps2, gamma = self.alpha, self.beta, self.eps1, self.eps2, self.gamma
        dt, v_threshold = self.dt, self.v_threshold

        def deriv(vv: float, ww: float, zz: float) -> tuple[float, float, float]:
            try:
                candidate = (
                    vv - vv * vv * vv / 3.0 - ww - zz + current,
                    eps1 * (vv - gamma * ww + alpha),
                    eps2 * (beta * (vv + 0.7) - zz),
                )
            except OverflowError as exc:
                raise FloatingPointError("Pernarowski Python batch derivative overflowed") from exc
            if not all(math.isfinite(value) for value in candidate):
                raise FloatingPointError("Pernarowski Python batch derivative became non-finite")
            return candidate

        spikes = 0
        for t in range(n_steps):
            v_prev = v
            dv1, dw1, dz1 = deriv(v, w, z)
            dv2, dw2, dz2 = deriv(v + 0.5 * dt * dv1, w + 0.5 * dt * dw1, z + 0.5 * dt * dz1)
            dv3, dw3, dz3 = deriv(v + 0.5 * dt * dv2, w + 0.5 * dt * dw2, z + 0.5 * dt * dz2)
            dv4, dw4, dz4 = deriv(v + dt * dv3, w + dt * dw3, z + dt * dz3)
            candidate = (
                v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0,
                w + dt * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4) / 6.0,
                z + dt * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4) / 6.0,
            )
            if not all(math.isfinite(value) for value in candidate):
                raise FloatingPointError("Pernarowski Python batch candidate became non-finite")
            v, w, z = candidate
            trace[t] = v
            if v >= v_threshold and v_prev < v_threshold:
                spikes += 1
        return trace, spikes, v, w, z

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, vf, wf, zf = _rust_simulate(
            self.v,
            self.w,
            self.z,
            self.alpha,
            self.beta,
            self.eps1,
            self.eps2,
            self.gamma,
            self.dt,
            self.v_threshold,
            n_steps,
            current,
        )
        return (
            np.asarray(trace_list, dtype=np.float64),
            int(spikes),
            float(vf),
            float(wf),
            float(zf),
        )

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _julia_module is not None
        try:
            result = _julia_module.simulate_trace(
                float(self.v),
                float(self.w),
                float(self.z),
                float(self.alpha),
                float(self.beta),
                float(self.eps1),
                float(self.eps2),
                float(self.gamma),
                float(self.dt),
                float(self.v_threshold),
                int(n_steps),
                float(current),
            )
        except Exception as exc:
            raise FloatingPointError(
                "Pernarowski Julia batch rejected an invalid candidate"
            ) from exc
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.vf), float(result.wf), float(result.zf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 3, dtype=np.float64, order="C")
        spikes = _go_lib.pernarowski_simulate_c(
            ctypes.c_double(self.v),
            ctypes.c_double(self.w),
            ctypes.c_double(self.z),
            ctypes.c_double(self.alpha),
            ctypes.c_double(self.beta),
            ctypes.c_double(self.eps1),
            ctypes.c_double(self.eps2),
            ctypes.c_double(self.gamma),
            ctypes.c_double(self.dt),
            ctypes.c_double(self.v_threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        wf = float(trace[n_steps + 1])
        zf = float(trace[n_steps + 2])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, wf, zf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 3, dtype=np.float64, order="C")
        spikes = _mojo_lib.pernarowski_simulate_c(
            float(self.v),
            float(self.w),
            float(self.z),
            float(self.alpha),
            float(self.beta),
            float(self.eps1),
            float(self.eps2),
            float(self.gamma),
            float(self.dt),
            float(self.v_threshold),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        vf = float(trace[n_steps]) if n_steps > 0 else self.v
        wf = float(trace[n_steps + 1])
        zf = float(trace[n_steps + 2])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), vf, wf, zf

    def reset(self) -> None:
        self.v, self.w, self.z = -1.0, 0.0, 0.0
