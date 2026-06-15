# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz et al. 2007 / Tanaka — piecewise-linear bursting map

"""Ibarz-Tanaka piecewise-linear bursting map neuron.

A two-dimensional discrete recurrence: a fast variable ``x`` driven by a
two-piece map (a subthreshold rational branch and a linear spiking branch),
modulated by a slowly drifting recovery variable ``y``, with an explicit reset
once ``x`` crosses threshold. It produces spiking and bursting without any ODE
integration.

    x(n+1) = alpha / (1 - x(n)) + y(n) + I    if x(n) <= 0
           = alpha + beta * x(n) + y(n) + I    otherwise
    y(n+1) = y(n) - mu * (x(n) + 1) + mu * sigma
    if x(n+1) >= x_threshold:  x(n+1) = x_reset   (spike)

Each step is exact floating-point arithmetic — one division, additions and
multiplications, no transcendental functions — so the N-step ``simulate``
accelerators (Rust, Julia, Go) reproduce the NumPy reference bit-for-bit.

Reference: Ibarz, B., Casado, J.M. & Sanjuán, M.A.F. (2011). "Map-based models
in neuronal dynamics." Phys. Rep. 501:1-74.
"""

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
# A single `step` is trivial, but an N-step simulation is a sequential
# recurrence (each step depends on the previous) that does not vectorise, so a
# compiled inner loop genuinely beats Python. The polyglot chain (Rust PyO3,
# Julia juliacall, Go cgo, Mojo FFI) accelerates `simulate` and reproduces the
# NumPy reference to the last bit.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_ibarz_tanaka_map_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "julia", "neurons", "ibarz_tanaka_map.jl")
    )
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.IbarzTanakaMapAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "ibarz_tanaka_map", "libibarz.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "ibarz_tanaka_map_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 8 + [
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libibarz.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "ibarz_tanaka_map_simulate_c", None)
    if fn is None:
        return False
    # 8 float params + n_steps + current + trace addr; returns spikes.
    fn.argtypes = [ctypes.c_double] * 8 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class IbarzTanakaMapNeuron:
    """Ibarz et al. 2007 / Tanaka — piecewise-linear bursting map.

    x(n+1) = f(x(n)) + y(n) + I
    y(n+1) = y(n) - mu*(x(n) + 1) + mu*sigma

    f(x) = alpha/(1-x)       if x <= 0
         = alpha + beta*x     if 0 < x < alpha+beta (spiking)
    Reset x -> x_reset when x >= x_threshold.

    Reference: Ibarz, B. et al. (2011). Phys. Rep. 501:1–74.
    """

    x: float = -1.0
    y: float = -2.5
    alpha: float = 3.65
    beta: float = 0.25
    mu: float = 0.0005
    sigma: float = -1.6
    x_threshold: float = 3.0
    x_reset: float = -1.0

    def _f(self, x: float) -> float:
        if x <= 0.0:
            return self.alpha / (1.0 - x)
        return self.alpha + self.beta * x

    def step(self, current: float) -> int:
        x_new = self._f(self.x) + self.y + current
        y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma
        self.x = x_new
        self.y = y_new
        if self.x >= self.x_threshold:
            self.x = self.x_reset
            return 1
        return 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the fast variable ``x`` after step ``t`` (already reset
        to ``x_reset`` on a spiking step); ``spikes`` counts the threshold
        crossings. The instance state ``(x, y)`` is advanced to the final step.
        Every backend reproduces the pure-NumPy reference bit-for-bit (Rust,
        Julia, Go) or to a documented per-step ULP bound (Mojo).
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust Ibarz-Tanaka backend requested but the engine wheel lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError(
                "Julia Ibarz-Tanaka backend requested but juliacall/.jl is unavailable."
            )
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Ibarz-Tanaka backend requested but libibarz.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/ibarz_tanaka_map && go build "
                "-buildmode=c-shared -o libibarz.so ibarz_tanaka_map.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Ibarz-Tanaka backend requested but libibarz.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libibarz.so ibarz_tanaka_map.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, xf, yf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, xf, yf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, xf, yf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, xf, yf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, xf, yf = self._simulate_python(n_steps, current)
        self.x, self.y = xf, yf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        x, y = self.x, self.y
        alpha, beta, mu, sigma = self.alpha, self.beta, self.mu, self.sigma
        thr, reset = self.x_threshold, self.x_reset
        spikes = 0
        for t in range(n_steps):
            if x <= 0.0:
                f = alpha / (1.0 - x)
            else:
                f = alpha + beta * x
            x_new = f + y + current
            y_new = y - mu * (x + 1.0) + mu * sigma
            y = y_new
            if x_new >= thr:
                x = reset
                spikes += 1
            else:
                x = x_new
            trace[t] = x
        return trace, spikes, x, y

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, xf, yf = _rust_simulate(
            self.x,
            self.y,
            self.alpha,
            self.beta,
            self.mu,
            self.sigma,
            self.x_threshold,
            self.x_reset,
            n_steps,
            current,
        )
        return np.asarray(trace_list, dtype=np.float64), int(spikes), float(xf), float(yf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.x),
            float(self.y),
            float(self.alpha),
            float(self.beta),
            float(self.mu),
            float(self.sigma),
            float(self.x_threshold),
            float(self.x_reset),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.xf), float(result.yf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _go_lib.ibarz_tanaka_map_simulate_c(
            ctypes.c_double(self.x),
            ctypes.c_double(self.y),
            ctypes.c_double(self.alpha),
            ctypes.c_double(self.beta),
            ctypes.c_double(self.mu),
            ctypes.c_double(self.sigma),
            ctypes.c_double(self.x_threshold),
            ctypes.c_double(self.x_reset),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        yf = float(trace[n_steps + 1]) if n_steps > 0 else self.y
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _mojo_lib.ibarz_tanaka_map_simulate_c(
            float(self.x),
            float(self.y),
            float(self.alpha),
            float(self.beta),
            float(self.mu),
            float(self.sigma),
            float(self.x_threshold),
            float(self.x_reset),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        yf = float(trace[n_steps + 1]) if n_steps > 0 else self.y
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf

    def reset(self) -> None:
        self.x = -1.0
        self.y = -2.5
