# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exponential integrate-and-fire neuron

"""Profile-explicit exponential integrate-and-fire dynamics and dispatch.

Both profiles implement Fourcaud-Trocmé et al. (2003), Equations 6 and 10,
after dividing the current balance by the leak conductance. The source factory
uses the paper's ``-30 mV`` simulation handoff, fitted ``1.7 ms`` refractory
duration, and a sub-``0.02 ms`` RK2 specialization. The zero-argument
constructor preserves the historical SC ``+30 mV`` candidate-first RK4/Q32.32
compatibility recurrence. ``v_threshold`` is distinct from the soft
exponential threshold ``v_rh`` in both profiles.
"""

from __future__ import annotations

import importlib as _importlib
import math
import os as _os
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


def _load_engine_expif() -> Any:
    """Return the checked Rust complete-batch entrypoint."""
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.expif_simulate_complete


try:
    _EngineExpIFSimulateFn = _load_engine_expif()
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineExpIFSimulateFn = None
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
    simulate = getattr(library, "expif_simulate_complete_c", None)
    if simulate is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 10 + [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    simulate.restype = ctypes.c_int64
    compatibility = getattr(library, "expif_simulate_c", None)
    if compatibility is not None:
        compatibility.argtypes = [ctypes.c_double] * 10 + [
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        compatibility.restype = ctypes.c_int64
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
    simulate = getattr(library, "expif_simulate_complete_c", None)
    if simulate is None:
        return False
    simulate.argtypes = [ctypes.c_double] * 10 + [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    simulate.restype = ctypes.c_int64
    compatibility = getattr(library, "expif_simulate_c", None)
    if compatibility is not None:
        compatibility.argtypes = [ctypes.c_double] * 10 + [
            ctypes.c_int64,
            ctypes.c_double,
            ctypes.c_int64,
        ]
        compatibility.restype = ctypes.c_int64
    _mojo_lib = library
    _HAS_MOJO = True
    return True


@dataclass
class ExpIFNeuron:
    """Profile-explicit Fourcaud-Trocmé exponential integrate-and-fire neuron.

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
    profile:
        ``"fourcaud_trocme_2003"`` selects the deterministic source
        specialization; ``"sc_rk4"`` retains the historical SC recurrence.

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
    profile: Literal["sc_rk4", "fourcaud_trocme_2003"] = "sc_rk4"

    def __post_init__(self) -> None:
        if self.profile not in {"sc_rk4", "fourcaud_trocme_2003"}:
            raise ValueError(f"unsupported ExpIF profile: {self.profile!r}")
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
        if self.profile == "fourcaud_trocme_2003":
            source_fit = {
                "v_rest": -65.0,
                "v_reset": -68.0,
                "v_threshold": -30.0,
                "v_rh": -59.9,
                "delta_t": 3.48,
                "tau": 10.0,
                "refractory_period": 1.7,
            }
            if any(float(getattr(self, name)) != value for name, value in source_fit.items()):
                raise ValueError("fourcaud_trocme_2003 profile requires the fitted source values")
            if self.dt >= 0.02:
                raise ValueError("fourcaud_trocme_2003 profile requires dt < 0.02 ms")

    @classmethod
    def fourcaud_trocme_2003(cls, *, dt: float = 0.01) -> ExpIFNeuron:
        """Return the fitted source protocol's deterministic zero-noise profile.

        The paper specifies stochastic second-order Runge-Kutta below the
        ``-30 mV`` handoff, analytical integration of the exponential-only
        tail above it, a ``1.7 ms`` refractory interval, and only constrains
        the numerical step to be below ``0.02 ms``.  ``dt=0.01 ms`` is the
        maintained converged specialization; it is not claimed as a verbatim
        source parameter.
        """
        return cls(
            v=-65.0,
            v_rest=-65.0,
            v_reset=-68.0,
            v_threshold=-30.0,
            v_rh=-59.9,
            delta_t=3.48,
            tau=10.0,
            dt=dt,
            refractory_period=1.7,
            refractory_remaining=0.0,
            profile="fourcaud_trocme_2003",
        )

    @classmethod
    def sc_rk4_compatibility(cls) -> ExpIFNeuron:
        """Return the historical SC candidate-first RK4/Q32.32 contract."""
        return cls()

    def analytical_tail_ms(self) -> float:
        """Return the source approximation from the handoff to divergence."""
        return self.tau * math.exp(-(self.v_threshold - self.v_rh) / self.delta_t)

    def _validate_runtime_state(self) -> None:
        """Reject corrupt runtime state before any mutation."""
        if not math.isfinite(self.v) or self.v >= self.v_threshold:
            raise ValueError("runtime voltage state must be finite and below v_threshold")
        if not math.isfinite(self.refractory_remaining) or not (
            0.0 <= self.refractory_remaining <= self.refractory_period
        ):
            raise ValueError("runtime refractory state is outside its valid interval")

    def _rhs(self, v: float, current: float) -> float:
        """Return the EIF voltage derivative at one Runge-Kutta stage."""
        bounded_v = min(v, self.v_threshold)
        try:
            exp_term = self.delta_t * math.exp((bounded_v - self.v_rh) / self.delta_t)
        except OverflowError as exc:
            raise ValueError("ExpIF exponential term must remain finite") from exc
        rhs = (-(bounded_v - self.v_rest) + exp_term + current) / self.tau
        if not math.isfinite(rhs):
            raise ValueError("ExpIF derivative must remain finite")
        return rhs

    def _rk4_candidate(self, current: float) -> float:
        """Return one candidate-first classical RK4 voltage update."""
        k1 = self._rhs(self.v, current)
        k2 = self._rhs(self.v + 0.5 * self.dt * k1, current)
        k3 = self._rhs(self.v + 0.5 * self.dt * k2, current)
        k4 = self._rhs(self.v + self.dt * k3, current)
        return self.v + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk2_candidate(self, current: float) -> float:
        """Return the deterministic Heun specialization of the source RK2."""
        k1 = self._rhs(self.v, current)
        predictor = self.v + self.dt * k1
        k2 = self._rhs(predictor, current)
        return self.v + 0.5 * self.dt * (k1 + k2)

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

        if self.profile == "fourcaud_trocme_2003":
            next_v = self._rk2_candidate(current)
        else:
            next_v = self._rk4_candidate(current)
        if not math.isfinite(next_v):
            method = "source RK2" if self.profile == "fourcaud_trocme_2003" else "SC RK4"
            raise ValueError(f"ExpIF update ({method}) must remain finite")
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
        voltage, _refractory, events = self.simulate_complete(n_steps, current, backend)
        return voltage, int(np.sum(events, dtype=np.int64))

    def simulate_complete(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
    ]:
        """Return aligned post-step voltage, refractory, and event traces.

        Every backend runs the complete batch against a candidate state.  The
        receiver advances only after packet shape, finiteness, event-domain,
        and final-state checks succeed, so a late failure is mutation-free.
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
            elif _HAS_RUST and _EngineExpIFSimulateFn is not None:
                selected = "rust"
            else:
                selected = "python"

        if selected == "rust":
            if not _HAS_RUST or _EngineExpIFSimulateFn is None:
                raise RuntimeError(
                    "Rust ExpIF backend requested but sc_neurocore_engine is unavailable."
                )
            packet = self._simulate_rust_complete(n_steps, current)
        elif selected == "julia":
            if not _ensure_julia_loaded():
                raise RuntimeError(
                    "Julia ExpIF backend requested but juliacall or the ExpIF module is unavailable."
                )
            packet = self._simulate_julia_complete(n_steps, current)
        elif selected == "go":
            if not _ensure_go_loaded():
                raise RuntimeError(
                    "Go ExpIF backend requested but libexpif.so is not built; run "
                    "go build -buildmode=c-shared -o libexpif.so expif.go in "
                    "accel/go/neurons/expif."
                )
            packet = self._simulate_go_complete(n_steps, current)
        elif selected == "mojo":
            if not _ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo ExpIF backend requested but libexpif.so is not built; run "
                    "mojo build --emit shared-lib -o libexpif.so expif.mojo in "
                    "accel/mojo/kernels."
                )
            packet = self._simulate_mojo_complete(n_steps, current)
        else:
            packet = self._simulate_python_complete(n_steps, current)

        voltage, refractory, events, final_state = self._validated_complete_packet(packet, n_steps)
        self.v, self.refractory_remaining = final_state
        return voltage, refractory, events

    @staticmethod
    def _validated_complete_packet(
        packet: tuple[object, object, object, tuple[float, float]],
        n_steps: int,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
        tuple[float, float],
    ]:
        raw_voltage, raw_refractory, raw_events, final_state = packet
        voltage = np.ascontiguousarray(np.asarray(raw_voltage, dtype=np.float64))
        refractory = np.ascontiguousarray(np.asarray(raw_refractory, dtype=np.float64))
        event_values = np.asarray(raw_events)
        if voltage.shape != (n_steps,) or refractory.shape != (n_steps,):
            raise FloatingPointError("ExpIF backend returned malformed state trace shapes.")
        if event_values.shape != (n_steps,):
            raise FloatingPointError("ExpIF backend returned a malformed event trace shape.")
        if not np.all(np.isfinite(voltage)) or not np.all(np.isfinite(refractory)):
            raise FloatingPointError("ExpIF backend returned non-finite trace state.")
        if not np.all((event_values == 0) | (event_values == 1)):
            raise FloatingPointError("ExpIF backend returned events outside the binary domain.")
        events = np.ascontiguousarray(event_values, dtype=np.uint8)
        final_v, final_refractory = map(float, final_state)
        if not math.isfinite(final_v) or not math.isfinite(final_refractory):
            raise FloatingPointError("ExpIF backend returned a non-finite final state.")
        if n_steps and (final_v != float(voltage[-1]) or final_refractory != float(refractory[-1])):
            raise FloatingPointError("ExpIF backend final state disagrees with its trace packet.")
        return voltage, refractory, events, (final_v, final_refractory)

    def _simulate_python_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        candidate = replace(self)
        voltage = np.empty(n_steps, dtype=np.float64)
        refractory = np.empty(n_steps, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        for index in range(n_steps):
            events[index] = candidate.step(current)
            voltage[index] = candidate.v
            refractory[index] = candidate.refractory_remaining
        return voltage, refractory, events, (candidate.v, candidate.refractory_remaining)

    def _simulate_rust_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        assert _EngineExpIFSimulateFn is not None
        result = _EngineExpIFSimulateFn(
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
            self.profile == "fourcaud_trocme_2003",
            int(n_steps),
            float(current),
        )
        return result[0], result[1], result[2], (float(result[3]), float(result[4]))

    def _simulate_julia_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        assert _julia_module is not None
        result = _julia_module.simulate_complete(
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
            self.profile == "fourcaud_trocme_2003",
            int(n_steps),
            float(current),
        )
        return (
            result.voltage,
            result.refractory,
            result.events,
            (float(result.vf), float(result.rf)),
        )

    def _simulate_go_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        assert _go_lib is not None
        import ctypes

        voltage = np.empty(n_steps + 1, dtype=np.float64)
        refractory = np.empty(n_steps + 1, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        spikes = int(
            _go_lib.expif_simulate_complete_c(
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
                int(self.profile == "fourcaud_trocme_2003"),
                n_steps,
                current,
                voltage.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                refractory.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            )
        )
        if spikes < 0:
            raise FloatingPointError("Go ExpIF kernel rejected the simulation contract.")
        if spikes != int(np.sum(events, dtype=np.int64)):
            raise FloatingPointError("Go ExpIF event count disagrees with its event trace.")
        return (
            voltage[:n_steps],
            refractory[:n_steps],
            events,
            (float(voltage[n_steps]), float(refractory[n_steps])),
        )

    def _simulate_mojo_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, object, tuple[float, float]]:
        assert _mojo_lib is not None
        voltage = np.empty(n_steps + 1, dtype=np.float64)
        refractory = np.empty(n_steps + 1, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        spikes = int(
            _mojo_lib.expif_simulate_complete_c(
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
                int(self.profile == "fourcaud_trocme_2003"),
                n_steps,
                current,
                int(voltage.ctypes.data),
                int(refractory.ctypes.data),
                int(events.ctypes.data),
            )
        )
        if spikes < 0:
            raise FloatingPointError("Mojo ExpIF kernel rejected the simulation contract.")
        if spikes != int(np.sum(events, dtype=np.int64)):
            raise FloatingPointError("Mojo ExpIF event count disagrees with its event trace.")
        return (
            voltage[:n_steps],
            refractory[:n_steps],
            events,
            (float(voltage[n_steps]), float(refractory[n_steps])),
        )

    def reset(self) -> None:
        """Restore resting voltage and clear any refractory hold."""
        self.v = self.v_rest
        self.refractory_remaining = 0.0
