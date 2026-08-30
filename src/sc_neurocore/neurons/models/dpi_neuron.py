# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Published current-mode DPI neuron circuit equations

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import dpi_neuron as _backends

_DPI_CONTRACT_FIELDS = (
    "i_mem",
    "i_ahp",
    "refractory_time",
    "i_threshold",
    "i_reset",
    "i_rest",
    "i_tau",
    "i_g",
    "i_tau_ahp",
    "i_ga",
    "i_spike",
    "i_0",
    "kappa",
    "alpha",
    "tau",
    "tau_ahp",
    "refractory_period",
    "dt",
)

_DPIState = tuple[float, float, float]
_DPICompleteRaw = tuple[object, object, object, object, _DPIState]
_DPICompletePacket = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
]


class _FullContractRunner(Protocol):
    """Callable contract shared by the configurable native backends."""

    def __call__(
        self,
        i_mem: float,
        i_ahp: float,
        refractory_time: float,
        i_threshold: float,
        i_reset: float,
        i_rest: float,
        i_tau: float,
        i_g: float,
        i_tau_ahp: float,
        i_ga: float,
        i_spike: float,
        i_0: float,
        kappa: float,
        alpha: float,
        tau: float,
        tau_ahp: float,
        refractory_period: float,
        dt: float,
        n_steps: int,
        current: float,
    ) -> _DPICompleteRaw: ...


def _sigmoid(value: float) -> float:
    """Evaluate the logistic gate without overflowing either exponential branch."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


@dataclass
class DPINeuron:
    r"""Current-mode conductance-based DPI silicon neuron.

    This class advances the coupled subthreshold equations derived for the
    differential-pair-integrator circuit by Indiveri, Stefanini, and Chicca
    (2010), Eqs. (2)--(3):

    .. math::

       I_{fb} = I_0^{1/(\kappa+1)} I_{mem}^{\kappa/(\kappa+1)}
                \left[1 + e^{-\alpha(I_{mem}-I_{th})}\right]^{-1}

    .. math::

       \tau \dot I_{mem} = \frac{I_{mem}}{I_\tau}
       \left(\frac{I_{rest}+I_{inj}}{1 + I_{mem}/I_g}
       - I_\tau + I_{fb} - I_{ahp}\right)

    .. math::

       \tau_{ahp} \dot I_{ahp} = \frac{I_{ahp}}{I_{\tau ahp}}
       \left(\frac{I_{spk} r(t)}{1 + I_{ahp}/I_{ga}}
       - I_{\tau ahp}\right).

    ``r(t)`` is one for the programmable refractory pulse and zero otherwise.
    The paper identifies the circuit paths that generate spikes, reset the
    membrane, impose refractoriness, and drive adaptation. This maintained
    digital realisation adds dimensionless factory parameters, ``i_rest`` as a
    constant component of ``I_in``, simultaneous explicit-Euler integration,
    post-step level detection, and a subsequent-step pulse that holds
    ``i_mem`` at ``i_reset``. Those numerical choices are not values or a
    discrete recurrence printed by the paper.

    References
    ----------
    Indiveri, G., Stefanini, F., & Chicca, E. (2010). *Spike-based learning
    with a generalized integrate and fire silicon neuron*. ISCAS,
    doi:10.1109/ISCAS.2010.5536980.

    Indiveri, G. et al. (2011). *Neuromorphic Silicon Neuron Circuits*.
    Frontiers in Neuroscience 5:73, doi:10.3389/fnins.2011.00073.
    """

    i_mem: float = 0.01
    i_ahp: float = 0.01
    refractory_time: float = 0.0
    i_threshold: float = 1.0
    i_reset: float = 0.01
    i_rest: float = 0.1
    i_tau: float = 1.0
    i_g: float = 1.0
    i_tau_ahp: float = 0.1
    i_ga: float = 1.0
    i_spike: float = 5.0
    i_0: float = 0.01
    kappa: float = 0.7
    alpha: float = 10.0
    tau: float = 20.0
    tau_ahp: float = 100.0
    refractory_period: float = 2.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        self._validate_runtime_state()

    def _feedback_current(self, i_mem: float) -> float:
        """Return the publication's inverter positive-feedback current."""
        log_current = (math.log(self.i_0) + self.kappa * math.log(i_mem)) / (self.kappa + 1.0)
        gate = _sigmoid(self.alpha * (i_mem - self.i_threshold))
        return math.exp(log_current) * gate

    def _derivatives(self, current: float, *, spike_active: bool) -> tuple[float, float]:
        """Evaluate the coupled current-domain right-hand sides."""
        spike_current = self.i_spike if spike_active else 0.0
        d_i_ahp = (
            self.i_ahp
            / (self.tau_ahp * self.i_tau_ahp)
            * (spike_current / (1.0 + self.i_ahp / self.i_ga) - self.i_tau_ahp)
        )
        if spike_active:
            return 0.0, d_i_ahp

        i_fb = self._feedback_current(self.i_mem)
        d_i_mem = (
            self.i_mem
            / (self.tau * self.i_tau)
            * (
                (self.i_rest + current) / (1.0 + self.i_mem / self.i_g)
                - self.i_tau
                + i_fb
                - self.i_ahp
            )
        )
        return d_i_mem, d_i_ahp

    def step(self, current: float) -> int:
        """Advance one mutation-atomic Euler update of the published circuit."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        total_input = self.i_rest + current
        if not math.isfinite(total_input) or total_input < 0.0:
            raise ValueError("i_rest + current must be finite and non-negative")

        spike_active = self.refractory_time > 0.0
        try:
            d_i_mem, d_i_ahp = self._derivatives(current, spike_active=spike_active)
        except (OverflowError, ValueError) as exc:
            raise ValueError("DPI nonlinear current evaluation failed") from exc

        next_i_ahp = self.i_ahp + self.dt * d_i_ahp
        if spike_active:
            next_i_mem = self.i_reset
            next_refractory = max(0.0, self.refractory_time - self.dt)
            spiked = False
        else:
            candidate_i_mem = self.i_mem + self.dt * d_i_mem
            if not math.isfinite(candidate_i_mem) or candidate_i_mem <= 0.0:
                raise ValueError("DPI membrane Euler candidate left the physical domain")
            next_i_mem = candidate_i_mem
            next_refractory = 0.0
            spiked = next_i_mem >= self.i_threshold
            if spiked:
                next_i_mem = self.i_reset
                next_refractory = self.refractory_period

        candidates = (next_i_mem, next_i_ahp, next_refractory)
        if not all(math.isfinite(value) for value in candidates):
            raise ValueError("DPI Euler update must remain finite")
        if next_i_mem <= 0.0 or next_i_ahp < 0.0 or next_refractory < 0.0:
            raise ValueError("DPI Euler update left the physical current domain")

        self.i_mem = next_i_mem
        self.i_ahp = next_i_ahp
        self.refractory_time = next_refractory
        return int(spiked)

    def simulate(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Return membrane trace and aggregate count through one backend."""
        i_mem, _i_ahp, _refractory, events = self.simulate_complete(n_steps, current, backend)
        return i_mem, int(np.sum(events, dtype=np.int64))

    def simulate_complete(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> _DPICompletePacket:
        """Return aligned state/events and atomically commit the final state.

        Parameters
        ----------
        n_steps:
            Number of constant-input Euler updates.
        current:
            Finite injected component of the maintained input current.
        backend:
            ``python``, ``rust``, ``julia``, ``go``, ``mojo``, or ``auto``.

        Returns
        -------
        tuple of numpy.ndarray
            Contiguous post-step membrane, adaptation, refractory, and binary
            event traces with identical length.

        Raises
        ------
        ValueError
            If the input, state, parameters, or requested backend are invalid.
        RuntimeError
            If an explicitly requested native runtime is unavailable or emits
            a malformed packet.
        FloatingPointError
            If a native runtime rejects arithmetic or emits non-finite state.
        """
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        total_input = self.i_rest + current
        if not math.isfinite(total_input) or total_input < 0.0:
            raise ValueError("i_rest + current must be finite and non-negative")

        selected = backend
        if selected == "auto":
            if _backends.ensure_go_loaded():
                selected = "go"
            elif _backends.ensure_julia_loaded():
                selected = "julia"
            elif _backends.ensure_mojo_loaded():
                selected = "mojo"
            elif _backends._HAS_RUST:
                selected = "rust"
            else:
                selected = "python"

        if selected == "rust":
            if not _backends._HAS_RUST:
                raise RuntimeError(
                    "Rust DPI backend requested but sc_neurocore_engine is unavailable."
                )
            packet = self._simulate_full_contract(
                _backends.simulate_rust_complete, n_steps, current
            )
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia DPI backend requested but juliacall or the module is unavailable."
                )
            packet = self._simulate_full_contract(
                _backends.simulate_julia_complete, n_steps, current
            )
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go DPI backend requested but libdpi_neuron.so is not built; run "
                    "go build -buildmode=c-shared -o libdpi_neuron.so . "
                    "in accel/go/neurons/dpi_neuron."
                )
            packet = self._simulate_full_contract(_backends.simulate_go_complete, n_steps, current)
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo DPI backend requested but libdpi_neuron.so is not built; run "
                    "mojo build --emit shared-lib -o libdpi_neuron.so dpi_neuron.mojo "
                    "in accel/mojo/kernels."
                )
            packet = self._simulate_full_contract(
                _backends.simulate_mojo_complete, n_steps, current
            )
        else:
            packet = self._simulate_python_complete(n_steps, current)

        traces, state = self._validated_complete_packet(packet, n_steps)
        self.i_mem, self.i_ahp, self.refractory_time = state
        return traces

    def _simulate_full_contract(
        self,
        runner: object,
        n_steps: int,
        current: float,
    ) -> _DPICompleteRaw:
        """Pass every maintained state and parameter to a native runner."""
        native = cast(_FullContractRunner, runner)
        return native(
            self.i_mem,
            self.i_ahp,
            self.refractory_time,
            self.i_threshold,
            self.i_reset,
            self.i_rest,
            self.i_tau,
            self.i_g,
            self.i_tau_ahp,
            self.i_ga,
            self.i_spike,
            self.i_0,
            self.kappa,
            self.alpha,
            self.tau,
            self.tau_ahp,
            self.refractory_period,
            self.dt,
            n_steps,
            current,
        )

    def _simulate_python_complete(self, n_steps: int, current: float) -> _DPICompleteRaw:
        """Run the maintained recurrence against a staged instance."""
        candidate = copy(self)
        i_mem = np.empty(n_steps, dtype=np.float64)
        i_ahp = np.empty(n_steps, dtype=np.float64)
        refractory = np.empty(n_steps, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        for index in range(n_steps):
            events[index] = candidate.step(current)
            i_mem[index] = candidate.i_mem
            i_ahp[index] = candidate.i_ahp
            refractory[index] = candidate.refractory_time
        state = (candidate.i_mem, candidate.i_ahp, candidate.refractory_time)
        return i_mem, i_ahp, refractory, events, state

    @staticmethod
    def _validated_complete_packet(
        packet: _DPICompleteRaw,
        n_steps: int,
    ) -> tuple[_DPICompletePacket, _DPIState]:
        """Validate one native packet before caller-visible state mutation."""
        raw_i_mem, raw_i_ahp, raw_refractory, raw_events, raw_state = packet
        i_mem = np.ascontiguousarray(np.asarray(raw_i_mem, dtype=np.float64))
        i_ahp = np.ascontiguousarray(np.asarray(raw_i_ahp, dtype=np.float64))
        refractory = np.ascontiguousarray(np.asarray(raw_refractory, dtype=np.float64))
        event_values = np.asarray(raw_events)
        if any(values.shape != (n_steps,) for values in (i_mem, i_ahp, refractory, event_values)):
            raise RuntimeError("DPI backend returned an invalid packet shape")
        state = tuple(float(value) for value in raw_state)
        if len(state) != 3:
            raise RuntimeError("DPI backend returned an invalid final state")
        if not (
            np.all(np.isfinite(i_mem))
            and np.all(np.isfinite(i_ahp))
            and np.all(np.isfinite(refractory))
            and all(math.isfinite(value) for value in state)
        ):
            raise FloatingPointError("DPI backend returned non-finite state")
        if (
            np.any(i_mem <= 0.0)
            or np.any(i_ahp < 0.0)
            or np.any(refractory < 0.0)
            or state[0] <= 0.0
            or state[1] < 0.0
            or state[2] < 0.0
        ):
            raise RuntimeError("DPI backend returned state outside the physical domain")
        if not np.all((event_values == 0) | (event_values == 1)):
            raise RuntimeError("DPI backend returned non-binary events")
        events = np.ascontiguousarray(event_values, dtype=np.uint8)
        if n_steps and state != (
            float(i_mem[-1]),
            float(i_ahp[-1]),
            float(refractory[-1]),
        ):
            raise RuntimeError("DPI backend final state disagrees with its traces")
        return (i_mem, i_ahp, refractory, events), state

    def reset(self) -> None:
        """Restore the physical leakage-current baseline and clear the pulse."""
        self.i_mem = self.i_reset
        self.i_ahp = self.i_0
        self.refractory_time = 0.0

    def _validate_runtime_state(self) -> None:
        """Validate the physical current-domain state and circuit parameters."""
        values = tuple(float(getattr(self, name)) for name in _DPI_CONTRACT_FIELDS)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("DPI state and parameters must be finite")
        if self.i_mem <= 0.0:
            raise ValueError("i_mem must be positive")
        if self.i_ahp < 0.0:
            raise ValueError("i_ahp must be non-negative")
        if self.refractory_time < 0.0:
            raise ValueError("refractory_time must be non-negative")
        if self.i_threshold <= 0.0:
            raise ValueError("i_threshold must be positive")
        if self.i_reset <= 0.0 or self.i_reset >= self.i_threshold:
            raise ValueError("i_reset must be positive and below i_threshold")
        if self.i_rest < 0.0:
            raise ValueError("i_rest must be non-negative")
        positive = {
            "i_tau": self.i_tau,
            "i_g": self.i_g,
            "i_tau_ahp": self.i_tau_ahp,
            "i_ga": self.i_ga,
            "i_spike": self.i_spike,
            "i_0": self.i_0,
            "kappa": self.kappa,
            "alpha": self.alpha,
            "tau": self.tau,
            "tau_ahp": self.tau_ahp,
            "refractory_period": self.refractory_period,
            "dt": self.dt,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.refractory_period < self.dt:
            raise ValueError("refractory_period must be at least one timestep")
