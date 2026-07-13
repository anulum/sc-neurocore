# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exponential integrate-and-fire neuron

"""Source-bound exponential integrate-and-fire dynamics and dispatch.

The maintained recurrence implements Fourcaud-Trocmé et al. (2003),
Equations 6 and 10, after dividing the current balance by the leak
conductance. The finite ``v_threshold`` is a numerical spike cutoff, not the
soft exponential threshold ``v_rh``. The paper used ``+30 mV`` for its fitted
EIF simulations and warned that a low cutoff can collapse the model towards a
leaky integrate-and-fire response.
"""

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

_EngineExpIF = Any


def _load_engine_expif() -> type[Any]:
    """Return the Rust engine's factory-default ExpIF class."""
    engine = _importlib.import_module("sc_neurocore_engine")
    return cast(type[Any], engine.ExpIFNeuron)


try:
    _EngineExpIFCls: type[Any] | None = _load_engine_expif()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineExpIFCls = None
    _HAS_RUST = False

_julia_module: Any | None = None
_HAS_JULIA = False
_go_lib: Any | None = None
_HAS_GO = False
_mojo_lib: Any | None = None
_HAS_MOJO = False

_ACCEL_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "accel"))


def _ensure_julia_loaded() -> bool:
    """Load the executable ExpIF Julia module when its runtime is available."""
    global _julia_module, _HAS_JULIA
    if _julia_module is not None:
        return True
    import importlib.util as importlib_util

    if importlib_util.find_spec("juliacall") is None:
        return False
    source_path = _os.path.join(_ACCEL_ROOT, "julia", "neurons", "expif.jl")
    if not _os.path.isfile(source_path):
        return False
    try:
        juliacall = _importlib.import_module("juliacall")
        julia = juliacall.Main
        julia.include(source_path)
        _julia_module = julia.ExpifAccel
    except (ImportError, AttributeError, RuntimeError):
        return False
    _HAS_JULIA = True
    return True


def _ensure_go_loaded() -> bool:
    """Load the compiled ExpIF Go C-ABI bridge when it is available."""
    global _go_lib, _HAS_GO
    if _go_lib is not None:
        return True
    import ctypes

    library_path = _os.path.join(_ACCEL_ROOT, "go", "neurons", "expif", "libexpif.so")
    if not _os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "expif_simulate_c", None)
    if simulate is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 10 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    simulate.restype = ctypes.c_int64
    _go_lib = library
    _HAS_GO = True
    return True


def _ensure_mojo_loaded() -> bool:
    """Load the compiled ExpIF Mojo C ABI when it is available."""
    global _mojo_lib, _HAS_MOJO
    if _mojo_lib is not None:
        return True
    import ctypes

    library_path = _os.path.join(_ACCEL_ROOT, "mojo", "kernels", "libexpif.so")
    if not _os.path.isfile(library_path):
        return False
    try:
        library = ctypes.CDLL(library_path)
    except OSError:
        return False
    simulate = getattr(library, "expif_simulate_c", None)
    if simulate is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 10 + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
    ]
    simulate.restype = ctypes.c_int64
    _mojo_lib = library
    _HAS_MOJO = True
    return True


_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "v": -65.0,
    "v_rest": -65.0,
    "v_reset": -68.0,
    "v_threshold": 30.0,
    "v_rh": -59.9,
    "delta_t": 3.48,
    "tau": 10.0,
    "dt": 0.02,
    "refractory_period": 0.0,
    "refractory_remaining": 0.0,
}


@dataclass
class ExpIFNeuron:
    """Fourcaud-Trocmé exponential integrate-and-fire neuron.

    The deterministic voltage flow is

    ``tau * dv/dt = -(v - v_rest) + delta_t * exp((v - v_rh) / delta_t) + current``.

    ``v_rh`` is the soft threshold of the exponential current;
    ``v_threshold`` is the finite numerical cutoff at which a spike is emitted
    and ``v`` resets. Runge-Kutta stages are bounded at that event surface, so
    stages that overshoot the cutoff cannot evaluate an irrelevant divergent
    voltage. ``refractory_period=1.7`` reproduces the refractory duration used
    for the paper's fitted Wang-Buzsáki comparison; the zero default is the
    deterministic schema-to-RTL contract.

    Parameters
    ----------
    v:
        Initial membrane voltage in millivolts.
    v_rest:
        Leak reversal voltage in millivolts.
    v_reset:
        Post-spike reset voltage in millivolts.
    v_threshold:
        Finite numerical spike cutoff in millivolts.
    v_rh:
        Soft exponential threshold in millivolts.
    delta_t:
        Positive exponential slope factor in millivolts.
    tau:
        Positive membrane time constant in milliseconds.
    dt:
        Positive integration timestep in milliseconds.
    refractory_period:
        Non-negative post-spike hold duration in milliseconds.
    refractory_remaining:
        Non-negative runtime remainder of the refractory hold.

    References
    ----------
    Fourcaud-Trocmé, N., Hansel, D., van Vreeswijk, C. & Brunel, N.
    (2003). *Journal of Neuroscience*, 23(37), 11628–11640.
    doi:10.1523/JNEUROSCI.23-37-11628.2003.
    """

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -68.0
    v_threshold: float = 30.0
    v_rh: float = -59.9
    delta_t: float = 3.48
    tau: float = 10.0
    dt: float = 0.02
    refractory_period: float = 0.0
    refractory_remaining: float = 0.0

    def __post_init__(self) -> None:
        for field in ("v", "v_rest", "v_reset", "v_threshold", "v_rh"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("delta_t", "tau", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        for field in ("refractory_period", "refractory_remaining"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
        if self.v_threshold <= self.v_rh:
            raise ValueError("v_threshold must exceed the soft threshold v_rh")
        if self.v >= self.v_threshold:
            raise ValueError("v must start below v_threshold")
        if self.v_rest >= self.v_threshold or self.v_reset >= self.v_threshold:
            raise ValueError("v_rest and v_reset must remain below v_threshold")
        if self.refractory_remaining > self.refractory_period:
            raise ValueError("refractory_remaining cannot exceed refractory_period")

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust factory contract."""
        return all(
            float(getattr(self, name)) == expected
            for name, expected in _RUST_ENGINE_DEFAULTS.items()
        )

    def _validate_runtime_state(self) -> None:
        """Reject corrupt runtime state before any mutation."""
        if not math.isfinite(self.v) or self.v >= self.v_threshold:
            raise ValueError("runtime voltage state must be finite and below v_threshold")
        if not math.isfinite(self.refractory_remaining) or not (
            0.0 <= self.refractory_remaining <= self.refractory_period
        ):
            raise ValueError("runtime refractory state is outside its valid interval")

    def _rhs(self, v: float, current: float) -> float:
        """Return the source equation's voltage derivative at one RK4 stage."""
        bounded_v = min(v, self.v_threshold)
        try:
            exp_term = self.delta_t * math.exp((bounded_v - self.v_rh) / self.delta_t)
        except OverflowError as exc:
            raise ValueError("RK4 exponential term must remain finite") from exc
        rhs = (-(bounded_v - self.v_rest) + exp_term + current) / self.tau
        if not math.isfinite(rhs):
            raise ValueError("RK4 derivative must remain finite")
        return rhs

    def _rk4_candidate(self, current: float) -> float:
        """Return one candidate-first classical RK4 voltage update."""
        k1 = self._rhs(self.v, current)
        k2 = self._rhs(self.v + 0.5 * self.dt * k1, current)
        k3 = self._rhs(self.v + 0.5 * self.dt * k2, current)
        k4 = self._rhs(self.v + self.dt * k3, current)
        return self.v + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, current: float) -> int:
        """Advance one timestep and return ``1`` on a spike, otherwise ``0``.

        Parameters
        ----------
        current:
            Constant-current sample after normalisation by leak conductance.

        Returns
        -------
        int
            Binary spike event for this macro step.

        Raises
        ------
        ValueError
            If the input, runtime state, derivative, or candidate is invalid.
        """
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        if self.refractory_remaining > 0.0:
            self.refractory_remaining = max(0.0, self.refractory_remaining - self.dt)
            self.v = self.v_reset
            return 0

        next_v = self._rk4_candidate(current)
        if not math.isfinite(next_v):
            raise ValueError("RK4 update must remain finite")
        if next_v >= self.v_threshold:
            self.v = self.v_reset
            self.refractory_remaining = self.refractory_period
            return 1
        self.v = next_v
        return 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a sequential trace through Python, Rust, Julia, Go, or Mojo.

        ``trace[t]`` contains the post-step voltage, including post-reset and
        refractory-held samples. The instance commits the returned final voltage
        and refractory remainder only after a compiled backend succeeds.

        Parameters
        ----------
        n_steps:
            Non-negative number of sequential updates.
        current:
            Finite constant current for every update.
        backend:
            One of ``"auto"``, ``"python"``, ``"rust"``, ``"julia"``,
            ``"go"``, or ``"mojo"``. Auto uses the measured compiled order
            Julia, Go, Mojo, compatible Rust, then Python.

        Returns
        -------
        tuple[numpy.ndarray, int]
            Contiguous voltage trace and total spike count.

        Raises
        ------
        ValueError
            If the step count, current, or backend selector is invalid.
        RuntimeError
            If a requested compiled backend is unavailable or incompatible.
        FloatingPointError
            If a C-ABI backend rejects the numeric contract.
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        selected = backend
        if selected == "auto":
            if _ensure_julia_loaded():
                selected = "julia"
            elif _ensure_go_loaded():
                selected = "go"
            elif _ensure_mojo_loaded():
                selected = "mojo"
            elif _HAS_RUST and self._matches_rust_engine_contract():
                selected = "rust"
            else:
                selected = "python"

        if selected == "rust":
            if not _HAS_RUST or _EngineExpIFCls is None:
                raise RuntimeError(
                    "Rust ExpIF backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust ExpIF engine backend requires factory-default parameters and state."
                )
            trace, spikes, state = self._simulate_rust(n_steps, current)
        elif selected == "julia":
            if not _ensure_julia_loaded():
                raise RuntimeError(
                    "Julia ExpIF backend requested but juliacall or the ExpIF module is unavailable."
                )
            trace, spikes, state = self._simulate_julia(n_steps, current)
        elif selected == "go":
            if not _ensure_go_loaded():
                raise RuntimeError(
                    "Go ExpIF backend requested but libexpif.so is not built; run "
                    "go build -buildmode=c-shared -o libexpif.so expif.go in "
                    "accel/go/neurons/expif."
                )
            trace, spikes, state = self._simulate_go(n_steps, current)
        elif selected == "mojo":
            if not _ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo ExpIF backend requested but libexpif.so is not built; run "
                    "mojo build --emit shared-lib -o libexpif.so expif.mojo in "
                    "accel/mojo/kernels."
                )
            trace, spikes, state = self._simulate_mojo(n_steps, current)
        else:
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.v, self.refractory_remaining = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        """Run the local sequential recurrence."""
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for index in range(n_steps):
            spikes += self.step(current)
            trace[index] = self.v
        return trace, spikes, (self.v, self.refractory_remaining)

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        """Run the Rust engine under its factory-default contract."""
        assert _EngineExpIFCls is not None
        neuron = _EngineExpIFCls()
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for index in range(n_steps):
            spikes += int(neuron.step(float(current)))
            state = neuron.get_state()
            trace[index] = float(state["v"])
        final = neuron.get_state()
        return trace, spikes, (float(final["v"]), float(final["refractory_remaining"]))

    def _simulate_julia(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        """Run the Julia recurrence with the complete numeric contract."""
        assert _julia_module is not None
        result = _julia_module.simulate_trace(
            float(self.v),
            float(self.v_rest),
            float(self.v_reset),
            float(self.v_threshold),
            float(self.v_rh),
            float(self.delta_t),
            float(self.tau),
            float(self.dt),
            float(self.refractory_period),
            float(self.refractory_remaining),
            int(n_steps),
            float(current),
        )
        trace = np.ascontiguousarray(np.asarray(result.trace, dtype=np.float64))
        return trace, int(result.spikes), (float(result.vf), float(result.rf))

    def _simulate_go(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        """Run the Go service recurrence through its C ABI."""
        assert _go_lib is not None
        import ctypes

        output = np.empty(n_steps + 2, dtype=np.float64)
        spikes = int(
            _go_lib.expif_simulate_c(
                self.v,
                self.v_rest,
                self.v_reset,
                self.v_threshold,
                self.v_rh,
                self.delta_t,
                self.tau,
                self.dt,
                self.refractory_period,
                self.refractory_remaining,
                n_steps,
                current,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
        if spikes < 0:
            raise FloatingPointError("Go ExpIF kernel rejected the simulation contract.")
        return (
            np.ascontiguousarray(output[:n_steps]),
            spikes,
            (float(output[n_steps]), float(output[n_steps + 1])),
        )

    def _simulate_mojo(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, tuple[float, float]]:
        """Run the Mojo recurrence through its C ABI."""
        assert _mojo_lib is not None
        output = np.empty(n_steps + 2, dtype=np.float64)
        spikes = int(
            _mojo_lib.expif_simulate_c(
                self.v,
                self.v_rest,
                self.v_reset,
                self.v_threshold,
                self.v_rh,
                self.delta_t,
                self.tau,
                self.dt,
                self.refractory_period,
                self.refractory_remaining,
                n_steps,
                current,
                int(output.ctypes.data),
            )
        )
        if spikes < 0:
            raise FloatingPointError("Mojo ExpIF kernel rejected the simulation contract.")
        return (
            np.ascontiguousarray(output[:n_steps]),
            spikes,
            (float(output[n_steps]), float(output[n_steps + 1])),
        )

    def reset(self) -> None:
        """Restore resting voltage and clear any refractory hold."""
        self.v = self.v_rest
        self.refractory_remaining = 0.0
