# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev 2005 — 1D piecewise-monotone spiking map

"""Medvedev (2005) one-dimensional piecewise-monotone spiking map neuron.

A single fast variable ``x`` confined to the unit interval by a circle map: an
expanding tent-like branch pair followed by a fold back into ``[0, 1)``. The
expansion (``alpha > 1``) makes the map chaotic, and threshold crossings of
``x`` mark spikes.

    x(n+1) = (alpha * x(n) + I) mod 1            if x(n) < beta
           = (alpha * (1 - x(n)) + I) mod 1       otherwise

Each step is exact floating-point arithmetic — a multiply, an add and a
fold into ``[0, 1)`` (the ``floor``-based modulo is exact for unit divisor),
no transcendental functions — so the N-step ``simulate`` accelerators (Rust,
Julia, Go) reproduce the NumPy reference bit-for-bit.

Reference: Medvedev, G.S. (2005). "Reduction of a model of an excitable cell to
a one-dimensional map." Physica D 202:37-59.
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
# recurrence (each step depends on the previous, and a spike depends on the
# previous x) that does not vectorise, so a compiled inner loop genuinely beats
# Python. The polyglot chain (Rust PyO3, Julia juliacall, Go cgo, Mojo FFI)
# accelerates `simulate` and reproduces the NumPy reference to the last bit.

_RustSimulate = Callable[..., "tuple[list[float], int, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_medvedev_map_simulate  # type: ignore[no-any-return]


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
    jl_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "julia", "neurons", "medvedev_map.jl"))
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.MedvedevMapAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "medvedev_map", "libmedvedev.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "medvedev_map_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 4 + [
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libmedvedev.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "medvedev_map_simulate_c", None)
    if fn is None:
        return False
    # 4 float params + n_steps + current + trace addr; returns spikes.
    fn.argtypes = [ctypes.c_double] * 4 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class MedvedevMapNeuron:
    """Medvedev 2005 — 1D piecewise-monotone spiking map.

    Reference: Medvedev, G.S. (2005). Physica D 202:37–59.
    """

    x: float = 0.0
    alpha: float = 3.5
    beta: float = 0.5
    x_threshold: float = 0.9

    def step(self, current: float = 0.0) -> int:
        x_prev = self.x
        if self.x < self.beta:
            self.x = self.alpha * self.x + current
        else:
            self.x = self.alpha * (1.0 - self.x) + current
        self.x = self.x % 1.0
        return 1 if (self.x >= self.x_threshold and x_prev < self.x_threshold) else 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the fast variable ``x`` after step ``t`` (folded into
        ``[0, 1)``); ``spikes`` counts upward crossings of ``x_threshold``. The
        instance state ``x`` is advanced to the final step. Rust, Julia and Go
        reproduce the pure-NumPy reference bit-for-bit; Mojo agrees to a
        documented per-step ULP bound.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust Medvedev backend requested but the engine wheel lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia Medvedev backend requested but juliacall/.jl is unavailable.")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Medvedev backend requested but libmedvedev.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/medvedev_map && go build "
                "-buildmode=c-shared -o libmedvedev.so medvedev_map.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Medvedev backend requested but libmedvedev.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libmedvedev.so medvedev_map.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, xf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, xf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, xf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, xf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, xf = self._simulate_python(n_steps, current)
        self.x = xf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        x = self.x
        alpha, beta, thr = self.alpha, self.beta, self.x_threshold
        spikes = 0
        for t in range(n_steps):
            x_prev = x
            if x < beta:
                x = alpha * x + current
            else:
                x = alpha * (1.0 - x) + current
            x = x % 1.0
            trace[t] = x
            if x >= thr and x_prev < thr:
                spikes += 1
        return trace, spikes, x

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _rust_simulate is not None
        trace_list, spikes, xf = _rust_simulate(
            self.x, self.alpha, self.beta, self.x_threshold, n_steps, current
        )
        return np.asarray(trace_list, dtype=np.float64), int(spikes), float(xf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.x),
            float(self.alpha),
            float(self.beta),
            float(self.x_threshold),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.xf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 1, dtype=np.float64, order="C")
        spikes = _go_lib.medvedev_map_simulate_c(
            ctypes.c_double(self.x),
            ctypes.c_double(self.alpha),
            ctypes.c_double(self.beta),
            ctypes.c_double(self.x_threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 1, dtype=np.float64, order="C")
        spikes = _mojo_lib.medvedev_map_simulate_c(
            float(self.x),
            float(self.alpha),
            float(self.beta),
            float(self.x_threshold),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf

    def reset(self) -> None:
        self.x = 0.0
