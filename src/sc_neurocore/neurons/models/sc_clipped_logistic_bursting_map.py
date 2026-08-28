# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained clipped-logistic fast/slow bursting recurrence

"""Count-neutral retained two-state clipped-logistic bursting recurrence.

This project recurrence preserves the historical fast/slow map without a
whole-model publication attribution. Each step uses only finite arithmetic,
and the accelerated simulation paths preserve the historical trajectory.

    x(n+1) = clip(a*x(n)*(1 - x(n)) - y(n) + I, -2, 2)
    y(n+1) = y(n) + epsilon * (x(n) - sigma)

The source-faithful Cazelles-Courbage-Rabinovich identity lives in
``cazelles_map.py``.
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
# The single `step` is trivial, but an N-step simulation is a sequential
# recurrence (each step depends on the previous) that does not vectorise,
# so a compiled inner loop genuinely beats Python. The polyglot chain
# (Rust PyO3, Julia juliacall, Go cgo, Mojo FFI) accelerates `simulate`
# and reproduces the NumPy reference to the last bit.

_RustSimulate = Callable[..., "tuple[list[float], int, float, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_sc_clipped_logistic_bursting_map_simulate  # type: ignore[no-any-return]


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
        _os.path.join(_ACCEL_ROOT, "julia", "neurons", "sc_clipped_logistic_bursting_map.jl")
    )
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.SCClippedLogisticBurstingMapAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(
            _ACCEL_ROOT,
            "go",
            "neurons",
            "sc_clipped_logistic_bursting_map",
            "libsc_clipped_logistic_bursting_map.so",
        )
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "sc_clipped_logistic_bursting_map_simulate_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_double] * 6 + [
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

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libsc_clipped_logistic_bursting_map.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "sc_clipped_logistic_bursting_map_simulate_c", None)
    if fn is None:
        return False
    # 6 float params + n_steps + current addr + trace addr; returns spikes
    fn.argtypes = [ctypes.c_double] * 6 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class SCClippedLogisticBurstingMapNeuron:
    """Retained project-defined two-state clipped-logistic bursting recurrence.

    x(n+1) = f(x(n)) - y(n) + I
    y(n+1) = y(n) + epsilon * (x(n) - sigma)

    f(x) = a*x*(1 - x)    (logistic-like fast dynamics)

    Bursting arises from slow y modulation of fast x.

    This is a project-defined recurrence without whole-model publication
    attribution.
    """

    x: float = 0.1
    y: float = 0.0
    a: float = 3.8
    epsilon: float = 0.01
    sigma: float = 0.5
    x_threshold: float = 0.9

    def step(self, current: float) -> int:
        f = self.a * self.x * (1.0 - self.x)
        x_new = f - self.y + current
        y_new = self.y + self.epsilon * (self.x - self.sigma)
        self.x = min(2.0, max(-2.0, x_new))
        self.y = y_new
        return 1 if self.x >= self.x_threshold else 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the membrane variable ``x`` after step ``t``; ``spikes``
        counts the steps whose ``x`` crossed ``x_threshold``. The instance state
        ``(x, y)`` is advanced to the final step. Every backend reproduces the
        pure-NumPy reference bit-for-bit.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError(
                "Rust SC clipped-logistic backend requested but the engine wheel lacks it."
            )
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError(
                "Julia SC clipped-logistic backend requested but juliacall/.jl is unavailable."
            )
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go SC clipped-logistic backend requested but libsc_clipped_logistic_bursting_map.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/sc_clipped_logistic_bursting_map && go build "
                "-buildmode=c-shared -o libsc_clipped_logistic_bursting_map.so sc_clipped_logistic_bursting_map.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo SC clipped-logistic backend requested but libsc_clipped_logistic_bursting_map.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libsc_clipped_logistic_bursting_map.so sc_clipped_logistic_bursting_map.mojo`."
            )

        params = (self.x, self.y, self.a, self.epsilon, self.sigma, self.x_threshold)
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
        _ = params
        self.x, self.y = xf, yf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        x, y = self.x, self.y
        spikes = 0
        for t in range(n_steps):
            f = self.a * x * (1.0 - x)
            x_new = f - y + current
            y_new = y + self.epsilon * (x - self.sigma)
            x = min(2.0, max(-2.0, x_new))
            y = y_new
            trace[t] = x
            if x >= self.x_threshold:
                spikes += 1
        return trace, spikes, x, y

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _rust_simulate is not None
        trace_list, spikes, xf, yf = _rust_simulate(
            self.x, self.y, self.a, self.epsilon, self.sigma, self.x_threshold, n_steps, current
        )
        return np.asarray(trace_list, dtype=np.float64), int(spikes), float(xf), float(yf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.x),
            float(self.y),
            float(self.a),
            float(self.epsilon),
            float(self.sigma),
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
        spikes = _go_lib.sc_clipped_logistic_bursting_map_simulate_c(
            ctypes.c_double(self.x),
            ctypes.c_double(self.y),
            ctypes.c_double(self.a),
            ctypes.c_double(self.epsilon),
            ctypes.c_double(self.sigma),
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
        spikes = _mojo_lib.sc_clipped_logistic_bursting_map_simulate_c(
            float(self.x),
            float(self.y),
            float(self.a),
            float(self.epsilon),
            float(self.sigma),
            float(self.x_threshold),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        xf = float(trace[n_steps]) if n_steps > 0 else self.x
        yf = float(trace[n_steps + 1])
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), xf, yf

    def reset(self) -> None:
        self.x = 0.1
        self.y = 0.0
