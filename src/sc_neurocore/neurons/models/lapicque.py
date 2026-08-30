# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lapicque 1907 polarization model + preserved SC LIF profile

"""Source-explicit Lapicque polarization dynamics and SC compatibility.

Lapicque's 1907 paper models nerve excitation as the first attainment of a
polarization threshold in a leaky-capacitor circuit. It does not define a
repetitive spiking neuron or an automatic post-event reset. The source profile
therefore latches its first excitation and never resets implicitly.

The historical SC-NeuroCore exact-flow, hard-reset LIF recurrence remains the
zero-argument compatibility profile. It is preserved deliberately, but is not
attributed as a verbatim source equation.
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import lapicque as _backends

LapicqueProfile = Literal["sc_lif", "lapicque_1907"]


@dataclass
class LapicqueNeuron:
    """Lapicque 1907 polarization threshold with an explicit SC LIF profile.

    For the source profile, with source voltage ``V``, series resistance
    ``R``, membrane/polarization resistance ``rho``, capacitance ``K``, and
    polarization ``v``, Lapicque derives

    ``K dv/dt = (V - v) / R - v / rho``.

    A constant-voltage pulse has the exact flow

    ``v(t+h) = v_inf + (v(t)-v_inf) exp(-h/beta)``,

    where ``v_inf = V*rho/(R+rho)`` and ``beta = K*R*rho/(R+rho)``.
    The first threshold attainment emits one latched excitation. There is no
    source-defined automatic reset.

    ``LapicqueNeuron()`` preserves the historical SC exact-flow LIF recurrence
    ``tau*dv/dt = -(v-v_rest) + resistance*current`` with hard reset.
    ``LapicqueNeuron.lapicque_1907()`` constructs the count-bearing source
    profile. Its normalized defaults preserve the paper's ``R > rho`` regime
    and ``beta=1 ms``; they are maintained reproducibility choices rather than
    claimed experimental constants.

    Reference
    ---------
    Lapicque, L. (1907). *Journal de Physiologie et de Pathologie Generale*,
    9, 620-635. English translation: doi:10.1007/s00422-007-0189-6.
    """

    # Shared dynamic state and observation threshold.
    v: float = 0.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_threshold: float = 1.0
    dt: float = 1.0

    # Preserved SC hard-reset LIF parameters.
    tau: float = 20.0
    resistance: float = 1.0

    # Lapicque 1907 circuit parameters. These fields are active only for the
    # source profile and retain the paper's distinct physical roles.
    capacitance: float = 1.1
    series_resistance: float = 10.0
    polarization_resistance: float = 1.0
    excited: bool = False
    profile: LapicqueProfile = "sc_lif"

    def __post_init__(self) -> None:
        self._validate_configuration()
        self._validate_runtime_state()

    @classmethod
    def lapicque_1907(
        cls,
        *,
        v: float = 0.0,
        v_threshold: float = 1.0,
        capacitance: float = 1.1,
        series_resistance: float = 10.0,
        polarization_resistance: float = 1.0,
        dt: float = 0.01,
        excited: bool = False,
    ) -> LapicqueNeuron:
        """Construct the source-equation polarization-threshold profile.

        The defaults are a normalized reproducibility point with
        ``series_resistance / polarization_resistance = 10`` and
        ``beta = 1 ms``. They are not presented as a parameter set measured by
        Lapicque. Input samples to :meth:`step` are source voltages, not injected
        membrane currents.
        """
        return cls(
            v=v,
            v_threshold=v_threshold,
            dt=dt,
            capacitance=capacitance,
            series_resistance=series_resistance,
            polarization_resistance=polarization_resistance,
            excited=excited,
            profile="lapicque_1907",
        )

    @classmethod
    def sc_lif_compatibility(
        cls,
        *,
        v: float = 0.0,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
        v_threshold: float = 1.0,
        tau: float = 20.0,
        resistance: float = 1.0,
        dt: float = 1.0,
    ) -> LapicqueNeuron:
        """Construct the preserved hard-reset SC exact-flow LIF profile."""
        return cls(
            v=v,
            v_rest=v_rest,
            v_reset=v_reset,
            v_threshold=v_threshold,
            tau=tau,
            resistance=resistance,
            dt=dt,
            profile="sc_lif",
        )

    @property
    def source_beta(self) -> float:
        """Return Lapicque's physical time constant ``K R rho/(R+rho)``."""
        return (
            self.capacitance
            * self.series_resistance
            * self.polarization_resistance
            / (self.series_resistance + self.polarization_resistance)
        )

    @property
    def source_alpha(self) -> float:
        """Return the infinite-duration threshold voltage ``v*(R+rho)/rho``."""
        return (
            self.v_threshold
            * (self.series_resistance + self.polarization_resistance)
            / self.polarization_resistance
        )

    def source_threshold_voltage(self, duration: float) -> float:
        """Return the source voltage required to reach threshold in ``duration``.

        This is Lapicque's strength-duration equation
        ``V = alpha / (1-exp(-duration/beta))``.
        """
        if self.profile != "lapicque_1907":
            raise RuntimeError("source_threshold_voltage requires the lapicque_1907 profile")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        return self.source_alpha / -math.expm1(-duration / self.source_beta)

    def _validate_configuration(self) -> None:
        if self.profile not in ("sc_lif", "lapicque_1907"):
            raise ValueError(f"unsupported Lapicque profile: {self.profile!r}")
        if not isinstance(self.excited, bool):
            raise ValueError("excited must be a boolean")
        if not math.isfinite(self.v_threshold) or self.v_threshold <= 0.0:
            raise ValueError("v_threshold must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        if self.profile == "lapicque_1907":
            for field in ("capacitance", "series_resistance", "polarization_resistance"):
                value = getattr(self, field)
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{field} must be finite and positive")
            return

        for field in ("v_rest", "v_reset"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v_threshold <= self.v_rest:
            raise ValueError("v_threshold must be greater than v_rest")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")
        for field in ("tau", "resistance"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if self.excited:
            raise ValueError("the SC LIF compatibility profile cannot start latched")

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.v):
            raise ValueError("runtime polarization/voltage state must be finite")
        if self.profile == "sc_lif" and self.v >= self.v_threshold:
            raise ValueError("v must be below v_threshold for the SC LIF profile")
        if self.profile == "lapicque_1907" and not self.excited and self.v >= self.v_threshold:
            raise ValueError("an unlatched source profile must start below threshold")

    def _source_candidate(self, source_voltage: float) -> float:
        total_resistance = self.series_resistance + self.polarization_resistance
        v_inf = source_voltage * self.polarization_resistance / total_resistance
        decay = math.exp(-self.dt / self.source_beta)
        candidate = v_inf + (self.v - v_inf) * decay
        if not math.isfinite(v_inf) or not math.isfinite(decay) or not math.isfinite(candidate):
            raise ValueError("Lapicque polarization candidate must remain finite")
        return candidate

    def _sc_lif_candidate(self, current: float) -> float:
        v_inf = self.v_rest + self.resistance * current
        decay = math.exp(-self.dt / self.tau)
        candidate = v_inf + (self.v - v_inf) * decay
        if not math.isfinite(v_inf) or not math.isfinite(decay) or not math.isfinite(candidate):
            raise ValueError("SC Lapicque LIF voltage candidate must remain finite")
        return candidate

    def step(self, drive: float) -> int:
        """Advance one exact constant-drive step and return a binary event.

        ``drive`` is source voltage for ``lapicque_1907`` and injected current
        for ``sc_lif``. The source event latches once and never triggers an
        implicit reset. The SC compatibility event performs the historical hard
        reset.
        """
        if not math.isfinite(drive):
            raise ValueError("drive/current/source voltage must be finite")
        self._validate_configuration()
        self._validate_runtime_state()

        if self.profile == "lapicque_1907":
            next_v = self._source_candidate(drive)
            event = int(not self.excited and next_v >= self.v_threshold)
            self.v = next_v
            if event:
                self.excited = True
            return event

        next_v = self._sc_lif_candidate(drive)
        if next_v >= self.v_threshold:
            self.v = self.v_reset
            return 1
        self.v = next_v
        return 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a complete batch and return its state trace and event count.

        ``current`` retains the historical public parameter name. It is a source
        voltage in the source profile and injected current in the SC profile.
        Use :meth:`simulate_complete` when per-step event custody is required.
        """
        voltage, events = self.simulate_complete(n_steps, current, backend)
        return voltage, int(np.sum(events, dtype=np.int64))

    def simulate_complete(
        self,
        n_steps: int,
        drive: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        """Return aligned post-step state and event traces, committing atomically."""
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(drive):
            raise ValueError("drive/current/source voltage must be finite")
        self._validate_configuration()
        self._validate_runtime_state()

        selected = backend
        if selected == "auto":
            if _backends.ensure_mojo_loaded():
                selected = "mojo"
            elif _backends.ensure_julia_loaded():
                selected = "julia"
            elif _backends.ensure_go_loaded():
                selected = "go"
            elif _backends._HAS_RUST:
                selected = "rust"
            else:
                selected = "python"

        arguments = (
            self.v,
            self.v_rest,
            self.v_reset,
            self.v_threshold,
            self.tau,
            self.resistance,
            self.dt,
            self.capacitance,
            self.series_resistance,
            self.polarization_resistance,
            self.excited,
            self.profile == "lapicque_1907",
            n_steps,
            drive,
        )
        if selected == "rust":
            if not _backends._HAS_RUST:
                raise RuntimeError(
                    "Rust Lapicque backend requested but sc_neurocore_engine is unavailable."
                )
            packet = _backends.simulate_rust_complete(*arguments)
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia Lapicque backend requested but juliacall or the module is unavailable."
                )
            packet = _backends.simulate_julia_complete(*arguments)
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go Lapicque backend requested but liblapicque.so is not built; run "
                    "go build -buildmode=c-shared -o liblapicque.so lapicque.go in "
                    "accel/go/neurons/lapicque."
                )
            packet = _backends.simulate_go_complete(*arguments)
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo Lapicque backend requested but liblapicque.so is not built; run "
                    "mojo build --emit shared-lib -o liblapicque.so lapicque.mojo in "
                    "accel/mojo/kernels."
                )
            packet = _backends.simulate_mojo_complete(*arguments)
        else:
            packet = self._simulate_python_complete(n_steps, drive)

        voltage, events, final_v, final_excited = self._validated_complete_packet(packet, n_steps)
        self.v = final_v
        self.excited = final_excited
        return voltage, events

    def _simulate_python_complete(
        self, n_steps: int, drive: float
    ) -> tuple[object, object, float, bool]:
        candidate = copy(self)
        voltage = np.empty(n_steps, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        for index in range(n_steps):
            events[index] = candidate.step(drive)
            voltage[index] = candidate.v
        return voltage, events, candidate.v, candidate.excited

    @staticmethod
    def _validated_complete_packet(
        packet: tuple[object, object, float, bool], n_steps: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], float, bool]:
        raw_voltage, raw_events, raw_final_v, raw_final_excited = packet
        voltage = np.ascontiguousarray(np.asarray(raw_voltage, dtype=np.float64))
        event_values = np.asarray(raw_events)
        if voltage.shape != (n_steps,):
            raise FloatingPointError("Lapicque backend returned a malformed state trace shape.")
        if event_values.shape != (n_steps,):
            raise FloatingPointError("Lapicque backend returned a malformed event trace shape.")
        if not np.all(np.isfinite(voltage)):
            raise FloatingPointError("Lapicque backend returned non-finite state.")
        if not np.all((event_values == 0) | (event_values == 1)):
            raise FloatingPointError("Lapicque backend returned events outside the binary domain.")
        if not math.isfinite(raw_final_v) or not isinstance(raw_final_excited, bool):
            raise FloatingPointError("Lapicque backend returned an invalid final state.")
        if n_steps and raw_final_v != float(voltage[-1]):
            raise FloatingPointError(
                "Lapicque backend final state disagrees with its trace packet."
            )
        events = np.ascontiguousarray(event_values, dtype=np.uint8)
        return voltage, events, float(raw_final_v), raw_final_excited

    def reset(self) -> None:
        """Re-arm the experiment or restore the SC membrane to rest.

        For the source profile this is an explicit protocol operation between
        pulses, not an automatic post-excitation rule from the 1907 paper.
        """
        self.v = 0.0 if self.profile == "lapicque_1907" else self.v_rest
        self.excited = False


class SCLapicqueLIFNeuron(LapicqueNeuron):
    """Count-neutral preserved exact-flow, hard-reset SC LIF identity.

    This explicit name prevents the later repetitive reset convention from
    being mistaken for the complete Lapicque 1907 source experiment.
    """

    def __init__(
        self,
        v: float = 0.0,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
        v_threshold: float = 1.0,
        tau: float = 20.0,
        resistance: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        super().__init__(
            v=v,
            v_rest=v_rest,
            v_reset=v_reset,
            v_threshold=v_threshold,
            tau=tau,
            resistance=resistance,
            dt=dt,
            profile="sc_lif",
        )
