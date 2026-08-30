# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Naud-Gerstner perfect integrator + preserved SC profile

"""Source-explicit perfect-integrator dynamics and SC compatibility.

Naud and Gerstner define the perfect integrator by ``dV/dt = I(t)/C`` and a
strict threshold rule: reset when ``V(t) > V_T``.  For the piecewise-constant
input contract used here, the step update is the exact integral of that
equation, rather than an Euler approximation.

The historical SC-NeuroCore recurrence used an inclusive ``>=`` comparator.
It remains the zero-argument compatibility profile and is also exposed under a
count-neutral explicit class name so that it cannot be mistaken for the source
boundary convention.
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import perfect_integrator as _backends

PerfectIntegratorProfile = Literal["sc_inclusive", "naud_gerstner_2012"]


@dataclass
class PerfectIntegratorNeuron:
    """Perfect integrator with source and preserved SC threshold profiles.

    The source profile implements Naud and Gerstner's equations
    ``dV/dt = I(t)/C`` and ``V > V_T -> V_r``.  The exact held-current update is
    ``V(t+h) = V(t) + I*h/C``.  ``PerfectIntegratorNeuron()`` preserves the
    historical inclusive SC comparator; use :meth:`naud_gerstner_2012` for the
    count-bearing source profile.

    The normalized defaults are maintained reproducibility choices, not
    experimental measurements reported by the source.

    Reference
    ---------
    Naud, R. and Gerstner, W. (2012). *The Performance (and Limits) of Simple
    Neuron Models: Generalizations of the Leaky Integrate-and-Fire Model*,
    section 1.1. doi:10.1007/978-94-007-3858-4_6.
    """

    v: float = 0.0
    c_m: float = 1.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dt: float = 0.1
    profile: PerfectIntegratorProfile = "sc_inclusive"

    def __post_init__(self) -> None:
        self._validate_configuration()
        self._validate_runtime_state()

    @classmethod
    def naud_gerstner_2012(
        cls,
        *,
        v: float = 0.0,
        c_m: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        dt: float = 0.1,
    ) -> PerfectIntegratorNeuron:
        """Construct the source-equation profile with a strict threshold."""
        return cls(
            v=v,
            c_m=c_m,
            v_threshold=v_threshold,
            v_reset=v_reset,
            dt=dt,
            profile="naud_gerstner_2012",
        )

    @classmethod
    def sc_inclusive_compatibility(
        cls,
        *,
        v: float = 0.0,
        c_m: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        dt: float = 0.1,
    ) -> PerfectIntegratorNeuron:
        """Construct the preserved inclusive-threshold SC profile."""
        return cls(
            v=v,
            c_m=c_m,
            v_threshold=v_threshold,
            v_reset=v_reset,
            dt=dt,
            profile="sc_inclusive",
        )

    def _validate_configuration(self) -> None:
        if self.profile not in ("sc_inclusive", "naud_gerstner_2012"):
            raise ValueError(f"unsupported perfect-integrator profile: {self.profile!r}")
        for field in ("v_threshold", "v_reset"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v_threshold <= self.v_reset:
            raise ValueError("v_threshold must be greater than v_reset")
        for field in ("c_m", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def _validate_runtime_state(self) -> None:
        if self.profile not in ("sc_inclusive", "naud_gerstner_2012"):
            raise ValueError("runtime profile is unsupported")
        for field in ("v", "c_m", "v_threshold", "v_reset", "dt"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"runtime {field} must be finite")
        if self.c_m <= 0.0 or self.dt <= 0.0:
            raise ValueError("runtime c_m and dt must be positive")
        if self.v_threshold <= self.v_reset:
            raise ValueError("runtime v_threshold must be greater than v_reset")
        if self.profile == "naud_gerstner_2012":
            if self.v > self.v_threshold:
                raise ValueError("source-profile v must not exceed v_threshold")
        elif self.v >= self.v_threshold:
            raise ValueError("SC-profile v must be below v_threshold")

    def step(self, current: float) -> int:
        """Advance one exact held-current step and return a binary event."""
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        increment = current * self.dt / self.c_m
        candidate = self.v + increment
        if not math.isfinite(increment) or not math.isfinite(candidate):
            raise ValueError("perfect-integrator voltage increment must remain finite")
        crossed = (
            candidate > self.v_threshold
            if self.profile == "naud_gerstner_2012"
            else candidate >= self.v_threshold
        )
        self.v = self.v_reset if crossed else candidate
        return int(crossed)

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a batch and return its post-step trace and event count."""
        voltage, events = self.simulate_complete(n_steps, current, backend)
        return voltage, int(np.sum(events, dtype=np.int64))

    def simulate_complete(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        """Return aligned state/event traces and commit valid state atomically."""
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
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
            self.c_m,
            self.v_threshold,
            self.v_reset,
            self.dt,
            self.profile == "naud_gerstner_2012",
            n_steps,
            current,
        )
        if selected == "rust":
            if not _backends._HAS_RUST:
                raise RuntimeError(
                    "Rust PerfectIntegrator backend requested but "
                    "sc_neurocore_engine is unavailable."
                )
            packet = _backends.simulate_rust_complete(*arguments)
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia PerfectIntegrator backend requested but juliacall or the "
                    "module is unavailable."
                )
            packet = _backends.simulate_julia_complete(*arguments)
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go PerfectIntegrator backend requested but "
                    "libperfect_integrator.so is not built."
                )
            packet = _backends.simulate_go_complete(*arguments)
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo PerfectIntegrator backend requested but "
                    "libperfect_integrator.so is not built."
                )
            packet = _backends.simulate_mojo_complete(*arguments)
        else:
            packet = self._simulate_python_complete(n_steps, current)

        voltage, events, final_v = self._validated_complete_packet(packet, n_steps)
        self.v = final_v
        return voltage, events

    def _simulate_python_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, float]:
        candidate = copy(self)
        voltage = np.empty(n_steps, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        for index in range(n_steps):
            events[index] = candidate.step(current)
            voltage[index] = candidate.v
        return voltage, events, candidate.v

    @staticmethod
    def _validated_complete_packet(
        packet: tuple[object, object, float], n_steps: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], float]:
        raw_voltage, raw_events, raw_final_v = packet
        voltage = np.ascontiguousarray(np.asarray(raw_voltage, dtype=np.float64))
        event_values = np.asarray(raw_events)
        if voltage.shape != (n_steps,):
            raise FloatingPointError(
                "PerfectIntegrator backend returned a malformed state trace shape."
            )
        if event_values.shape != (n_steps,):
            raise FloatingPointError(
                "PerfectIntegrator backend returned a malformed event trace shape."
            )
        if not np.all(np.isfinite(voltage)):
            raise FloatingPointError("PerfectIntegrator backend returned non-finite state.")
        if not np.all((event_values == 0) | (event_values == 1)):
            raise FloatingPointError(
                "PerfectIntegrator backend returned events outside the binary domain."
            )
        if not math.isfinite(raw_final_v):
            raise FloatingPointError("PerfectIntegrator backend returned invalid final state.")
        if n_steps and raw_final_v != float(voltage[-1]):
            raise FloatingPointError(
                "PerfectIntegrator backend final state disagrees with its trace packet."
            )
        events = np.ascontiguousarray(event_values, dtype=np.uint8)
        return voltage, events, float(raw_final_v)

    def reset(self) -> None:
        """Apply the source-defined reset operation explicitly."""
        self.v = self.v_reset


class SCInclusivePerfectIntegratorNeuron(PerfectIntegratorNeuron):
    """Count-neutral identity for the preserved inclusive SC recurrence."""

    def __init__(
        self,
        v: float = 0.0,
        c_m: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        dt: float = 0.1,
    ) -> None:
        super().__init__(
            v=v,
            c_m=c_m,
            v_threshold=v_threshold,
            v_reset=v_reset,
            dt=dt,
            profile="sc_inclusive",
        )
