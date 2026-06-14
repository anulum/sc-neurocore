# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hindmarsh-Rose 1984 — 3D chaotic bursting model

from __future__ import annotations

import importlib as _importlib
import os as _os
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import math

import numpy as np
import numpy.typing as npt

# ───────────────────────── backend detection ─────────────────────────
#
# `step` runs one RK4 update; an N-step `simulate` is a sequential recurrence
# (each step depends on the previous) that does not vectorise, so a compiled
# inner loop genuinely beats Python. The polyglot chain accelerates the RK4
# integrator (the production default). The Hindmarsh-Rose right-hand side is
# exact arithmetic (cubes/squares written as repeated multiplication — no
# transcendental functions), so Rust, Julia and Go reproduce the NumPy trace
# bit-for-bit even in the chaotic bursting regime. Mojo's FMA contraction makes
# its trace diverge once the dynamics are chaotic, so it is validated per step.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_hindmarsh_rose_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "hindmarsh_rose.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.HindmarshRoseAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "hindmarsh_rose", "libhr.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "hindmarsh_rose_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 9 + [
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libhr.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "hindmarsh_rose_simulate_c", None)
    if fn is None:
        return False
    # 9 float params + n_steps + current + trace addr; returns spikes.
    fn.argtypes = [ctypes.c_double] * 9 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class HindmarshRoseNeuron:
    """Hindmarsh-Rose 1984 — 3D chaotic bursting model.

    dx/dt = y - x³ + bx² - z + I
    dy/dt = 1 - 5x² - y
    dz/dt = r(s(x - x_rest) - z)

    Reference: Hindmarsh, J.L. & Rose, R.M. (1984). Proc. R. Soc. Lond. B 221:87–102.
    """

    x: float = -1.6
    y: float = -10.0
    z: float = 2.0
    b: float = 3.0
    r: float = 0.001
    s: float = 4.0
    x_rest: float = -1.6
    dt: float = 0.1
    x_threshold: float = 1.0
    integrator: Literal["rk4", "euler"] = "rk4"

    def __post_init__(self) -> None:
        if self.integrator not in {"rk4", "euler"}:
            raise ValueError("integrator must be 'rk4' or 'euler'")
        for name in ("x", "y", "z", "b", "r", "s", "x_rest", "dt", "x_threshold"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, float(value))
        for name in ("dt", "r", "s"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _derivatives(
        self, x: float, y: float, z: float, current: float
    ) -> tuple[float, float, float]:
        if not (
            math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(current)
        ):
            raise FloatingPointError("Hindmarsh-Rose derivative input became non-finite")
        try:
            # `x * x` / `x2 * x` (not `x**2` / `x**3`) so the powers are exact
            # IEEE multiplication and bit-identical to the Rust `x.powi(2)` /
            # `x.powi(3)`, Julia `x^2` / `x^3` and Go/Mojo `x*x` / `x*x*x` used
            # by the polyglot `simulate` backends.
            x2 = x * x
            x3 = x2 * x
            dx = y - x3 + self.b * x2 - z + current
            dy = 1.0 - 5.0 * x2 - y
            dz = self.r * (self.s * (x - self.x_rest) - z)
        except OverflowError as exc:
            raise FloatingPointError("Hindmarsh-Rose derivative overflowed") from exc
        if not (math.isfinite(dx) and math.isfinite(dy) and math.isfinite(dz)):
            raise FloatingPointError("Hindmarsh-Rose derivative became non-finite")
        return dx, dy, dz

    def _set_state(self, x: float, y: float, z: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise FloatingPointError("Hindmarsh-Rose state became non-finite")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def _step_euler(self, current: float) -> None:
        dx, dy, dz = self._derivatives(self.x, self.y, self.z, current)
        self._set_state(self.x + dx * self.dt, self.y + dy * self.dt, self.z + dz * self.dt)

    def _step_rk4(self, current: float) -> None:
        x0, y0, z0 = self.x, self.y, self.z
        dt = self.dt
        k1 = self._derivatives(x0, y0, z0, current)
        k2 = self._derivatives(
            x0 + 0.5 * dt * k1[0],
            y0 + 0.5 * dt * k1[1],
            z0 + 0.5 * dt * k1[2],
            current,
        )
        k3 = self._derivatives(
            x0 + 0.5 * dt * k2[0],
            y0 + 0.5 * dt * k2[1],
            z0 + 0.5 * dt * k2[2],
            current,
        )
        k4 = self._derivatives(x0 + dt * k3[0], y0 + dt * k3[1], z0 + dt * k3[2], current)
        self._set_state(
            x0 + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            y0 + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            z0 + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        )

    def step(self, current: float) -> int:
        if not isinstance(current, int | float) or not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        current = float(current)
        x_prev = self.x
        if self.integrator == "rk4":
            self._step_rk4(current)
        else:
            self._step_euler(current)
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` RK4 steps from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane variable ``x`` after step ``t``; ``spikes``
        counts upward crossings of ``x_threshold``. The instance state
        ``(x, y, z)`` is advanced to the final step.

        The compiled backends implement the **RK4** integrator (the production
        default); this method therefore requires ``integrator == "rk4"``. The
        right-hand side is exact arithmetic, so Rust, Julia and Go reproduce the
        pure-NumPy reference bit-for-bit even though the bursting dynamics are
        chaotic; Mojo's FMA contraction diverges under that chaos, so it is
        validated on the per-step ULP bound rather than the whole trace.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if not isinstance(current, int | float) or not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        current = float(current)
        if self.integrator != "rk4":
            raise ValueError(
                "simulate() accelerates the RK4 integrator only; this neuron uses "
                f"integrator={self.integrator!r}. Call step() in a loop, or set integrator='rk4'."
            )
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust Hindmarsh-Rose backend requested but the engine lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia Hindmarsh-Rose backend requested but it is unavailable.")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Hindmarsh-Rose backend requested but libhr.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/hindmarsh_rose && go build "
                "-buildmode=c-shared -o libhr.so hindmarsh_rose.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Hindmarsh-Rose backend requested but libhr.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libhr.so hindmarsh_rose.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, xf, yf, zf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, xf, yf, zf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, xf, yf, zf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, xf, yf, zf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, xf, yf, zf = self._simulate_python(n_steps, current)
        self.x, self.y, self.z = xf, yf, zf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        x, y, z = self.x, self.y, self.z
        b, r, s, x_rest, dt, thr = self.b, self.r, self.s, self.x_rest, self.dt, self.x_threshold
        dt6 = dt / 6.0
        spikes = 0

        def deriv(xx: float, yy: float, zz: float) -> tuple[float, float, float]:
            x2 = xx * xx
            x3 = x2 * xx
            return (
                yy - x3 + b * x2 - zz + current,
                1.0 - 5.0 * x2 - yy,
                r * (s * (xx - x_rest) - zz),
            )

        for t in range(n_steps):
            x_prev = x
            k1x, k1y, k1z = deriv(x, y, z)
            k2x, k2y, k2z = deriv(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z)
            k3x, k3y, k3z = deriv(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z)
            k4x, k4y, k4z = deriv(x + dt * k3x, y + dt * k3y, z + dt * k3z)
            x = x + dt6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            y = y + dt6 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
            z = z + dt6 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
            trace[t] = x
            if x >= thr and x_prev < thr:
                spikes += 1
        return trace, spikes, x, y, z

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, xf, yf, zf = _rust_simulate(
            self.x,
            self.y,
            self.z,
            self.b,
            self.r,
            self.s,
            self.x_rest,
            self.dt,
            self.x_threshold,
            n_steps,
            current,
        )
        return (
            np.asarray(trace_list, dtype=np.float64),
            int(spikes),
            float(xf),
            float(yf),
            float(zf),
        )

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.x),
            float(self.y),
            float(self.z),
            float(self.b),
            float(self.r),
            float(self.s),
            float(self.x_rest),
            float(self.dt),
            float(self.x_threshold),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.xf), float(result.yf), float(result.zf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 3, dtype=np.float64, order="C")
        spikes = _go_lib.hindmarsh_rose_simulate_c(
            ctypes.c_double(self.x),
            ctypes.c_double(self.y),
            ctypes.c_double(self.z),
            ctypes.c_double(self.b),
            ctypes.c_double(self.r),
            ctypes.c_double(self.s),
            ctypes.c_double(self.x_rest),
            ctypes.c_double(self.dt),
            ctypes.c_double(self.x_threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        yf = float(trace[n_steps + 1]) if n_steps > 0 else self.y
        zf = float(trace[n_steps + 2]) if n_steps > 0 else self.z
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf, zf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 3, dtype=np.float64, order="C")
        spikes = _mojo_lib.hindmarsh_rose_simulate_c(
            float(self.x),
            float(self.y),
            float(self.z),
            float(self.b),
            float(self.r),
            float(self.s),
            float(self.x_rest),
            float(self.dt),
            float(self.x_threshold),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        yf = float(trace[n_steps + 1]) if n_steps > 0 else self.y
        zf = float(trace[n_steps + 2]) if n_steps > 0 else self.z
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf, zf

    def reset(self) -> None:
        self.x = -1.6
        self.y = -10.0
        self.z = 2.0
