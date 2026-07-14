# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brette et al. 2007 conductance-based LIF benchmark

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import coba_lif as _backends

_COBAState = tuple[float, float, float, float]
_COBAResult = tuple[npt.NDArray[np.float64], int, _COBAState]


class _FullContractRunner(Protocol):
    """Callable contract shared by the configurable native backends."""

    def __call__(
        self,
        v: float,
        g_e: float,
        g_i: float,
        refractory_time: float,
        c_m: float,
        g_l: float,
        e_l: float,
        e_e: float,
        e_i: float,
        tau_e: float,
        tau_i: float,
        v_threshold: float,
        v_reset: float,
        refractory_period: float,
        dt: float,
        n_steps: int,
        current: float,
        delta_ge: float,
        delta_gi: float,
    ) -> _COBAResult: ...


@dataclass
class COBALIFNeuron:
    r"""Brette et al. conductance-based integrate-and-fire benchmark cell.

    C dV/dt = -g_L(V - E_L) - g_e(V - E_e) - g_i(V - E_i) + I
    dg_e/dt = -g_e / tau_e, dg_i/dt = -g_i / tau_i.

    Boundary conductance increments are applied before integration. Outside
    refractory periods, the full ``(v, g_e, g_i)`` candidate is advanced with
    coupled RK4. After a spike the voltage remains at ``v_reset`` for the
    source's 5 ms refractory interval while both conductances continue their
    RK4 decay. Every update is candidate-first and mutation-atomic.

    The continuous equations and factory defaults reproduce Benchmark 1 in
    Brette et al. (2007). RK4 is the maintained repository discretisation; the
    paper compared Euler, second-order Runge--Kutta, and spike-interpolated
    Euler implementations rather than prescribing RK4.

    References
    ----------
    Brette, R. et al. (2007). *Simulation of networks of spiking neurons: a
    review of tools and strategies*. Journal of Computational Neuroscience,
    23, 349--398. doi:10.1007/s10827-007-0038-6.
    """

    v: float = -60.0
    g_e: float = 0.0
    g_i: float = 0.0
    refractory_time: float = 0.0
    c_m: float = 200.0
    g_l: float = 10.0
    e_l: float = -60.0
    e_e: float = 0.0
    e_i: float = -80.0
    tau_e: float = 5.0
    tau_i: float = 10.0
    v_threshold: float = -50.0
    v_reset: float = -60.0
    refractory_period: float = 5.0
    dt: float = 0.1

    _V_MIN: ClassVar[float] = -200.0
    _V_MAX: ClassVar[float] = 100.0
    _G_MAX: ClassVar[float] = 1.0e9

    def __post_init__(self) -> None:
        self._validated_state()

    @staticmethod
    def _finite(value: float, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        value = cls._finite(value, name)
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _validated_state(self) -> _COBAState:
        v = self._finite(self.v, "v")
        if not self._V_MIN <= v <= self._V_MAX:
            raise ValueError("v outside COBA LIF safety envelope")
        g_e = self._nonnegative(self.g_e, "g_e")
        g_i = self._nonnegative(self.g_i, "g_i")
        if g_e > self._G_MAX or g_i > self._G_MAX:
            raise ValueError("conductance outside COBA LIF safety envelope")
        refractory_time = self._nonnegative(self.refractory_time, "refractory_time")

        self._positive(self.c_m, "c_m")
        self._nonnegative(self.g_l, "g_l")
        self._finite(self.e_l, "e_l")
        self._finite(self.e_e, "e_e")
        self._finite(self.e_i, "e_i")
        self._positive(self.tau_e, "tau_e")
        self._positive(self.tau_i, "tau_i")
        self._finite(self.v_threshold, "v_threshold")
        self._finite(self.v_reset, "v_reset")
        if not self._V_MIN <= self.v_reset <= self._V_MAX:
            raise ValueError("v_reset outside COBA LIF safety envelope")
        refractory_period = self._positive(self.refractory_period, "refractory_period")
        self._positive(self.dt, "dt")
        if refractory_period < self.dt:
            raise ValueError("refractory_period must be at least one timestep")
        if refractory_time > refractory_period:
            raise ValueError("refractory_time cannot exceed refractory_period")

        return v, g_e, g_i, refractory_time

    def step(self, current: float, delta_ge: float = 0.0, delta_gi: float = 0.0) -> int:
        """Advance one candidate-first RK4 timestep.

        Parameters
        ----------
        current:
            External drive current.
        delta_ge:
            Instantaneous excitatory conductance increment.
        delta_gi:
            Instantaneous inhibitory conductance increment.

        Returns
        -------
        int
            ``1`` when the RK4 voltage candidate crosses threshold, otherwise
            ``0``.

        Raises
        ------
        ValueError
            If the state, parameters, inputs, or candidate leave the maintained
            finite safety envelope. The stored state is unchanged on failure.
        """
        current = self._finite(current, "current")
        delta_ge = self._nonnegative(delta_ge, "delta_ge")
        delta_gi = self._nonnegative(delta_gi, "delta_gi")
        v, g_e, g_i, refractory_time = self._validated_state()

        g_e_pre = g_e + delta_ge
        g_i_pre = g_i + delta_gi
        if g_e_pre > self._G_MAX or g_i_pre > self._G_MAX:
            raise ValueError("conductance candidate outside COBA LIF safety envelope")

        if refractory_time > 0.0:
            g_e_candidate, g_i_candidate = self._conductance_candidates(g_e_pre, g_i_pre)
            v_candidate = self.v_reset
            refractory_candidate = (
                0.0 if refractory_time <= self.dt * (1.0 + 1.0e-12) else refractory_time - self.dt
            )
            spiked = False
        else:
            v_candidate, g_e_candidate, g_i_candidate = self._rk4_candidate(
                v, g_e_pre, g_i_pre, current
            )
            if not math.isfinite(v_candidate) or not self._V_MIN <= v_candidate <= self._V_MAX:
                raise ValueError("voltage candidate outside COBA LIF safety envelope")
            refractory_candidate = 0.0
            spiked = v_candidate >= self.v_threshold
            if spiked:
                v_candidate = self.v_reset
                refractory_candidate = self.refractory_period

        for value, name in (
            (v_candidate, "voltage candidate"),
            (g_e_candidate, "excitatory conductance candidate"),
            (g_i_candidate, "inhibitory conductance candidate"),
            (refractory_candidate, "refractory candidate"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not self._V_MIN <= v_candidate <= self._V_MAX:
            raise ValueError("voltage candidate outside COBA LIF safety envelope")
        if g_e_candidate < 0.0 or g_i_candidate < 0.0:
            raise ValueError("conductance candidate must remain non-negative")

        self.v = v_candidate
        self.g_e = g_e_candidate
        self.g_i = g_i_candidate
        self.refractory_time = refractory_candidate
        return int(spiked)

    def simulate(
        self,
        n_steps: int,
        current: float = 0.0,
        delta_ge: float = 0.0,
        delta_gi: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a constant-input trace through Python or a native backend.

        ``delta_ge`` and ``delta_gi`` are boundary increments applied on every
        simulated step. Event-driven callers can use :meth:`step` directly for
        arbitrary conductance-event schedules.

        Parameters
        ----------
        n_steps:
            Number of complete boundary-update and RK4 macro steps.
        current:
            Constant external drive current.
        delta_ge:
            Excitatory conductance increment applied before every macro step.
        delta_gi:
            Inhibitory conductance increment applied before every macro step.
        backend:
            ``"python"`` or one of the real native runtime names; ``"auto"``
            selects the fastest available backend established by the committed
            benchmark evidence.

        Returns
        -------
        tuple[numpy.ndarray, int]
            Post-step voltage trace and total spike count. The instance commits
            the returned final four-state tuple only after the backend succeeds.

        Raises
        ------
        ValueError
            If the requested contract is malformed or outside the maintained
            safety envelope.
        RuntimeError
            If an explicitly requested native runtime is unavailable.
        FloatingPointError
            If a compiled C ABI rejects the supplied contract.
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        current = self._finite(current, "current")
        delta_ge = self._nonnegative(delta_ge, "delta_ge")
        delta_gi = self._nonnegative(delta_gi, "delta_gi")
        self._validated_state()

        selected = backend
        if selected == "auto":
            # Controlled single-CPU evidence ranks the warmed production batch
            # lanes Julia > Rust > Mojo > Go for this coupled RK4 workload.
            if _backends.ensure_julia_loaded():
                selected = "julia"
            elif _backends._HAS_RUST:
                selected = "rust"
            elif _backends.ensure_mojo_loaded():
                selected = "mojo"
            elif _backends.ensure_go_loaded():
                selected = "go"
            else:
                selected = "python"

        if selected == "rust":
            if not _backends._HAS_RUST:
                raise RuntimeError(
                    "Rust COBA LIF backend requested but sc_neurocore_engine is unavailable."
                )
            result = self._simulate_full_contract(
                _backends.simulate_rust, n_steps, current, delta_ge, delta_gi
            )
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia COBA LIF backend requested but juliacall or the module is unavailable."
                )
            result = self._simulate_full_contract(
                _backends.simulate_julia, n_steps, current, delta_ge, delta_gi
            )
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError("Go COBA LIF backend requested but libcoba_lif.so is not built.")
            result = self._simulate_full_contract(
                _backends.simulate_go, n_steps, current, delta_ge, delta_gi
            )
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo COBA LIF backend requested but libcoba_lif.so is not built."
                )
            result = self._simulate_full_contract(
                _backends.simulate_mojo, n_steps, current, delta_ge, delta_gi
            )
        else:
            result = self._simulate_python(n_steps, current, delta_ge, delta_gi)

        trace, spikes, state = result
        self.v, self.g_e, self.g_i, self.refractory_time = state
        return trace, spikes

    def _simulate_full_contract(
        self,
        runner: object,
        n_steps: int,
        current: float,
        delta_ge: float,
        delta_gi: float,
    ) -> _COBAResult:
        """Pass every maintained state and parameter to a native runner."""
        native = cast(_FullContractRunner, runner)
        return native(
            self.v,
            self.g_e,
            self.g_i,
            self.refractory_time,
            self.c_m,
            self.g_l,
            self.e_l,
            self.e_e,
            self.e_i,
            self.tau_e,
            self.tau_i,
            self.v_threshold,
            self.v_reset,
            self.refractory_period,
            self.dt,
            n_steps,
            current,
            delta_ge,
            delta_gi,
        )

    def _simulate_python(
        self,
        n_steps: int,
        current: float,
        delta_ge: float,
        delta_gi: float,
    ) -> _COBAResult:
        """Run the maintained Python recurrence."""
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for index in range(n_steps):
            spikes += self.step(current, delta_ge, delta_gi)
            trace[index] = self.v
        return trace, spikes, (self.v, self.g_e, self.g_i, self.refractory_time)

    def reset(self) -> None:
        """Restore the membrane to leak reversal and clear conductances."""
        self.v = self.e_l
        self.g_e = 0.0
        self.g_i = 0.0
        self.refractory_time = 0.0

    def _conductance_candidates(self, g_e: float, g_i: float) -> tuple[float, float]:
        """Return RK4 decay candidates while the membrane is held at reset."""

        def decay(value: float, tau: float) -> float:
            k1 = -value / tau
            k2 = -(value + 0.5 * self.dt * k1) / tau
            k3 = -(value + 0.5 * self.dt * k2) / tau
            k4 = -(value + self.dt * k3) / tau
            return value + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return decay(g_e, self.tau_e), decay(g_i, self.tau_i)

    def _derivatives(
        self, v: float, g_e: float, g_i: float, current: float
    ) -> tuple[float, float, float]:
        i_syn = g_e * (v - self.e_e) + g_i * (v - self.e_i)
        dv = (-self.g_l * (v - self.e_l) - i_syn + current) / self.c_m
        dge = -g_e / self.tau_e
        dgi = -g_i / self.tau_i
        return dv, dge, dgi

    def _rk4_candidate(
        self, v: float, g_e: float, g_i: float, current: float
    ) -> tuple[float, float, float]:
        """Return the coupled RK4 candidate for ``(v, g_e, g_i)``."""
        k1v, k1e, k1i = self._derivatives(v, g_e, g_i, current)
        k2v, k2e, k2i = self._derivatives(
            v + 0.5 * self.dt * k1v,
            g_e + 0.5 * self.dt * k1e,
            g_i + 0.5 * self.dt * k1i,
            current,
        )
        k3v, k3e, k3i = self._derivatives(
            v + 0.5 * self.dt * k2v,
            g_e + 0.5 * self.dt * k2e,
            g_i + 0.5 * self.dt * k2i,
            current,
        )
        k4v, k4e, k4i = self._derivatives(
            v + self.dt * k3v,
            g_e + self.dt * k3e,
            g_i + self.dt * k3i,
            current,
        )
        return (
            v + (self.dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
            g_e + (self.dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
            g_i + (self.dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i),
        )
