# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Courbage, Nekorkin & Vdovin 2007 — piecewise-linear Lorenz-type map

"""Courbage-Nekorkin-Vdovin (2007) discontinuous two-dimensional spiking map.

Discrete-time phenomenological neuron built from a discrete FitzHugh-Nagumo
system plus a one-dimensional Lorenz-type map, with a Heaviside discontinuity at
``x = d`` that sets the excitation threshold (Courbage, Nekorkin & Vdovin,
Chaos 17:043109, 2007; arXiv:0712.2097, eqs. 3-5):

    x(n+1) = x(n) + F(x(n)) - y(n) - beta*H(x(n) - d) + I
    y(n+1) = y(n) + eps*(x(n) - J)

    F(x) = -m0*x          for x <= Jmin
           m1*(x - a)     for Jmin < x < Jmax
           -m0*(x - 1)    for x >= Jmax

    H(z) = 1 for z >= 0, else 0          (Heaviside step)
    Jmin = a*m1/(m0 + m1)                (continuity breakpoints of F)
    Jmax = (m0 + a*m1)/(m0 + m1)

``x`` is the fast membrane-like variable, ``y`` the slow recovery variable, and
``I`` (``current``) an injected external stimulus (``I = 0`` reproduces the
published autonomous map). The default parameters place the model in the
published chaotic spiking-bursting regime (Table, p20): ``m0 = 0.0864``,
``m1 = 0.65``, ``a = 0.2`` are the figure-1 values; ``d``, ``J``, ``beta`` lie
inside the ``B^+`` invariant-region triangle (eqs. 6, 9, 12) and ``eps < m0``.

Each step is exact floating-point arithmetic (additions, multiplications, one
division for the breakpoints, and a piecewise/Heaviside branch — no
transcendental functions), so the N-step ``simulate`` accelerators reproduce the
NumPy reference bit-for-bit on the bit-exact backends (Rust/Julia/Go). The map
is chaotic, so the FMA-fusing Mojo backend is validated per-step and on spike
counts rather than on the whole trace.
"""

from __future__ import annotations

import importlib as _importlib
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
# Julia juliacall, Go cgo, Mojo FFI) accelerates `simulate`.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_courage_nekorkin_map_simulate  # type: ignore[no-any-return]


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
        _os.path.join(_ACCEL_ROOT, "julia", "neurons", "courage_nekorkin_map.jl")
    )
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.CourageNekorkinMapAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(
            _ACCEL_ROOT, "go", "neurons", "courage_nekorkin_map", "libcourage.so"
        )
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "courage_nekorkin_map_simulate_c", None)
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libcourage.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "courage_nekorkin_map_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 10 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class CourageNekorkinMapNeuron:
    """Courbage-Nekorkin-Vdovin 2007 discontinuous 2D spiking-bursting map.

    x(n+1) = x(n) + F(x(n)) - y(n) - beta*H(x(n) - d) + I
    y(n+1) = y(n) + eps*(x(n) - J)

    with the piecewise-linear ``F`` and the Heaviside ``H`` of the module
    docstring. Defaults sit in the published chaotic spiking-bursting regime.

    Reference: Courbage, M., Nekorkin, V.I. & Vdovin, L.V. (2007).
    "Chaotic oscillations in a map-based model of neural activity."
    Chaos 17:043109 (arXiv:0712.2097), eqs. 3-6.
    """

    x: float = 0.0
    y: float = 0.0
    m0: float = 0.0864
    m1: float = 0.65
    a: float = 0.2
    d: float = 0.235
    j: float = 0.2
    beta: float = 0.085
    eps: float = 0.02
    x_threshold: float = 0.235

    def _breakpoints(self) -> tuple[float, float]:
        """Continuity breakpoints ``(Jmin, Jmax)`` of ``F`` (eq. 4)."""
        am1 = self.a * self.m1
        den = self.m0 + self.m1
        return am1 / den, (self.m0 + am1) / den

    def _f(self, x: float) -> float:
        """Piecewise-linear ``F(x)`` (Courbage et al. 2007, eq. 4)."""
        jmin, jmax = self._breakpoints()
        if x <= jmin:
            return -self.m0 * x
        if x < jmax:
            return self.m1 * (x - self.a)
        return -self.m0 * (x - 1.0)

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        jmin, jmax = self._breakpoints()
        if self.x <= jmin:
            fx = -self.m0 * self.x
        elif self.x < jmax:
            fx = self.m1 * (self.x - self.a)
        else:
            fx = -self.m0 * (self.x - 1.0)
        h = 1.0 if (self.x - self.d) >= 0.0 else 0.0
        x_new = self.x + fx - self.y - self.beta * h + current
        y_new = self.y + self.eps * (self.x - self.j)
        self.x = x_new
        self.y = y_new
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane variable ``x`` after step ``t``; ``spikes``
        counts the steps whose ``x`` crossed ``x_threshold`` upward. The instance
        state ``(x, y)`` is advanced to the final step. The Rust/Julia/Go
        backends reproduce the pure-NumPy reference bit-for-bit; the FMA-fusing
        Mojo backend is within a measured per-step ULP band with identical spike
        counts.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust Courbage backend requested but the engine wheel lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia Courbage backend requested but juliacall/.jl is unavailable.")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Courbage backend requested but libcourage.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/courage_nekorkin_map && go build "
                "-buildmode=c-shared -o libcourage.so courage_nekorkin_map.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Courbage backend requested but libcourage.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libcourage.so courage_nekorkin_map.mojo`."
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
        am1 = self.a * self.m1
        den = self.m0 + self.m1
        jmin = am1 / den
        jmax = (self.m0 + am1) / den
        spikes = 0
        for t in range(n_steps):
            x_prev = x
            if x <= jmin:
                fx = -self.m0 * x
            elif x < jmax:
                fx = self.m1 * (x - self.a)
            else:
                fx = -self.m0 * (x - 1.0)
            h = 1.0 if (x - self.d) >= 0.0 else 0.0
            x_new = x + fx - y - self.beta * h + current
            y_new = y + self.eps * (x - self.j)
            x = x_new
            y = y_new
            trace[t] = x
            if x >= self.x_threshold and x_prev < self.x_threshold:
                spikes += 1
        return trace, spikes, x, y

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, xf, yf = _rust_simulate(
            self.x,
            self.y,
            self.m0,
            self.m1,
            self.a,
            self.d,
            self.j,
            self.beta,
            self.eps,
            self.x_threshold,
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
            float(self.m0),
            float(self.m1),
            float(self.a),
            float(self.d),
            float(self.j),
            float(self.beta),
            float(self.eps),
            float(self.x_threshold),
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
        spikes = _go_lib.courage_nekorkin_map_simulate_c(
            ctypes.c_double(self.x),
            ctypes.c_double(self.y),
            ctypes.c_double(self.m0),
            ctypes.c_double(self.m1),
            ctypes.c_double(self.a),
            ctypes.c_double(self.d),
            ctypes.c_double(self.j),
            ctypes.c_double(self.beta),
            ctypes.c_double(self.eps),
            ctypes.c_double(self.x_threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        yf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 2, dtype=np.float64, order="C")
        spikes = _mojo_lib.courage_nekorkin_map_simulate_c(
            float(self.x),
            float(self.y),
            float(self.m0),
            float(self.m1),
            float(self.a),
            float(self.d),
            float(self.j),
            float(self.beta),
            float(self.eps),
            float(self.x_threshold),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        yf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf

    def reset(self) -> None:
        self.x = 0.0
        self.y = 0.0
