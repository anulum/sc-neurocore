# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ermentrout-Kopell Canonical Type I Map Neuron

"""Ermentrout-Kopell (1986) canonical Type I (theta neuron) map.

The canonical model for Type I (saddle-node-on-invariant-circle) excitability:
a single phase variable ``theta`` advances on a circle, and a spike is emitted
when ``theta`` crosses ``pi`` upward.

    theta(n+1) = (theta(n) + dt * [(1 - cos theta) + (1 + cos theta) * gain * I]) mod 2*pi

This is a phase oscillator, not a chaotic map (Lyapunov exponent 0), so per-step
floating-point differences do not amplify: the only transcendental is ``cos``.
On a shared libm, the Rust backend reproduces the NumPy reference bit-for-bit;
the Julia, Go and Mojo backends use their own ``cos`` and therefore agree to a
small, non-amplifying ULP bound, with identical spike counts.

Reference: Ermentrout, G.B. & Kopell, N. (1986). "Parabolic bursting in an
excitable system coupled with a slow oscillation." SIAM J. Appl. Math.
46(2):233-253.
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
# previous phase) that does not vectorise, so a compiled inner loop genuinely
# beats Python. The polyglot chain (Rust PyO3, Julia juliacall, Go cgo, Mojo
# FFI) accelerates `simulate`; because the theta neuron is non-chaotic, the
# transcendental `cos` differences across libm implementations stay bounded.

_RustSimulate = Callable[..., "tuple[list[float], int, float]"]


def _load_rust_simulate() -> _RustSimulate:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_ermentrout_kopell_map_simulate  # type: ignore[no-any-return]


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
        _os.path.join(_ACCEL_ROOT, "julia", "neurons", "ermentrout_kopell_map_neuron.jl")
    )
    if not _os.path.isfile(jl_path):
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl.include(jl_path)
    _julia_module = jl.ErmentroutKopellMapAccel
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.abspath(
        _os.path.join(_ACCEL_ROOT, "go", "neurons", "ermentrout_kopell_map", "libermentrout.so")
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "ermentrout_kopell_map_simulate_c", None)
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

    so_path = _os.path.abspath(_os.path.join(_ACCEL_ROOT, "mojo", "neurons", "libermentrout.so"))
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "ermentrout_kopell_map_simulate_c", None)
    if fn is None:
        return False
    # theta0, dt, gain, theta_threshold + n_steps + current + trace addr; returns spikes.
    fn.argtypes = [ctypes.c_double] * 4 + [ctypes.c_int64, ctypes.c_double, ctypes.c_int64]
    fn.restype = ctypes.c_int64
    _mojo_lib = lib
    _HAS_MOJO = True
    return True


@dataclass
class ErmentroutKopellMapNeuron:
    """Ermentrout-Kopell 1986 canonical Type I (theta neuron) map.

    The canonical model for Type I (saddle-node) excitability. Phase
    variable θ advances on a circle; spike occurs when θ crosses π.

    θ(n+1) = θ(n) + dt · [(1 - cos θ) + (1 + cos θ) · I]

    Reference: Ermentrout & Kopell (1986) SIAM J Appl Math 46:233–253.
    """

    theta: float = 0.0
    dt: float = 0.1
    gain: float = 1.0
    theta_threshold: float = math.pi

    def __post_init__(self) -> None:
        for name in ("theta", "dt", "gain", "theta_threshold"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    @staticmethod
    def _validate_theta(theta: float) -> float:
        value = float(theta)
        if not math.isfinite(value):
            raise FloatingPointError("Ermentrout-Kopell phase state must be finite")
        return value

    def step(self, current: float = 0.0) -> int:
        drive = float(current)
        if not math.isfinite(drive):
            raise ValueError("current must be finite")

        theta = self._validate_theta(self.theta)
        inp = self.gain * drive
        if not math.isfinite(inp):
            raise FloatingPointError("Ermentrout-Kopell input drive became non-finite")
        theta_prev = theta

        cos_theta = math.cos(theta)
        d_theta = (1.0 - cos_theta) + (1.0 + cos_theta) * inp
        theta_next = theta + self.dt * d_theta
        if not math.isfinite(d_theta) or not math.isfinite(theta_next):
            raise FloatingPointError("Ermentrout-Kopell candidate phase became non-finite")

        fired = 1 if theta_next >= self.theta_threshold and theta_prev < self.theta_threshold else 0
        two_pi = 2.0 * math.pi
        self.theta = theta_next % two_pi

        return fired

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` from the current state, returning ``(trace, spikes)``.

        ``trace[t]`` is the phase ``theta`` after step ``t`` (wrapped to
        ``[0, 2*pi)``); ``spikes`` counts upward crossings of
        ``theta_threshold``. The instance state ``theta`` is advanced to the
        final step. On a shared libm the Rust backend reproduces the NumPy
        reference bit-for-bit; Julia, Go and Mojo agree to a small,
        non-amplifying ULP bound with identical spike counts.
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if not math.isfinite(float(current)):
            raise ValueError("current must be finite")
        if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
            raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")

        if backend == "rust" and not _HAS_RUST:
            raise RuntimeError("Rust Ermentrout-Kopell backend requested but the engine lacks it.")
        if backend == "julia" and not _ensure_julia_loaded():
            raise RuntimeError("Julia Ermentrout-Kopell backend requested but it is unavailable.")
        if backend == "go" and not _ensure_go_loaded():
            raise RuntimeError(
                "Go Ermentrout-Kopell backend requested but libermentrout.so is not built; run "
                "`cd src/sc_neurocore/accel/go/neurons/ermentrout_kopell_map && go build "
                "-buildmode=c-shared -o libermentrout.so ermentrout_kopell_map.go`."
            )
        if backend == "mojo" and not _ensure_mojo_loaded():
            raise RuntimeError(
                "Mojo Ermentrout-Kopell backend requested but libermentrout.so is not built; run "
                "`cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib "
                "-o libermentrout.so ermentrout_kopell_map_neuron.mojo`."
            )

        if backend == "rust" or (backend == "auto" and _HAS_RUST):
            trace, spikes, tf = self._simulate_rust(n_steps, current)
        elif backend == "julia":
            trace, spikes, tf = self._simulate_julia(n_steps, current)
        elif backend == "go":
            trace, spikes, tf = self._simulate_go(n_steps, current)
        elif backend == "mojo":
            trace, spikes, tf = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, tf = self._simulate_python(n_steps, current)
        self.theta = tf
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        theta = self.theta
        dt, thr = self.dt, self.theta_threshold
        inp = self.gain * current
        two_pi = 2.0 * math.pi
        spikes = 0
        for t in range(n_steps):
            theta_prev = theta
            cos_theta = math.cos(theta)
            d_theta = (1.0 - cos_theta) + (1.0 + cos_theta) * inp
            theta_next = theta + dt * d_theta
            if theta_next >= thr and theta_prev < thr:
                spikes += 1
            theta = theta_next % two_pi
            trace[t] = theta
        return trace, spikes, theta

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _rust_simulate is not None
        trace_list, spikes, tf = _rust_simulate(
            self.theta, self.dt, self.gain, self.theta_threshold, n_steps, current
        )
        return np.asarray(trace_list, dtype=np.float64), int(spikes), float(tf)

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.theta),
            float(self.dt),
            float(self.gain),
            float(self.theta_threshold),
            int(n_steps),
            float(current),
        )
        trace = np.asarray(result.trace, dtype=np.float64)
        return trace, int(result.spikes), float(result.thetaf)

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _go_lib is not None
        import ctypes

        trace = np.zeros(n_steps + 1, dtype=np.float64, order="C")
        spikes = _go_lib.ermentrout_kopell_map_simulate_c(
            ctypes.c_double(self.theta),
            ctypes.c_double(self.dt),
            ctypes.c_double(self.gain),
            ctypes.c_double(self.theta_threshold),
            ctypes.c_int(n_steps),
            ctypes.c_double(current),
            trace.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        tf = float(trace[n_steps]) if n_steps > 0 else self.theta
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), tf

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _mojo_lib is not None
        trace = np.zeros(n_steps + 1, dtype=np.float64, order="C")
        spikes = _mojo_lib.ermentrout_kopell_map_simulate_c(
            float(self.theta),
            float(self.dt),
            float(self.gain),
            float(self.theta_threshold),
            int(n_steps),
            float(current),
            int(trace.ctypes.data),
        )
        tf = float(trace[n_steps]) if n_steps > 0 else self.theta
        return np.ascontiguousarray(trace[:n_steps]), int(spikes), tf

    def reset(self) -> None:
        self.theta = 0.0
