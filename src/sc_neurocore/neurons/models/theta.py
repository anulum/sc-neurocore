# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta neuron — canonical Type-I on the unit circle

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import theta as _backends


@dataclass
class ThetaNeuron:
    """Theta neuron — canonical Type-I on the unit circle.

    dθ/dt = (1 - cos θ) + (1 + cos θ) · I
    Spike when θ crosses π.
    Ermentrout & Kopell 1986.

    Reference: Ermentrout, G.B. & Kopell, N. (1986). SIAM J. Appl. Math. 46:233–253.

    This is the paper's constant-parameter equation (2.5), or equation (3.3)
    under a frozen slow drive. It is not the full coupled parabolic-bursting
    system. ``current`` is the source's dimensionless parameter ``a``.

    ``simulate_complete`` exposes aligned phase/event packets through the
    Python reference and all four compiled acceleration lanes.
    """

    theta: float = 0.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("theta must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        self.theta = self._wrap_phase(self.theta)

    @staticmethod
    def _wrap_phase(theta: float) -> float:
        return ((theta + math.pi) % (2.0 * math.pi)) - math.pi

    def _exact_candidate(self, current: float) -> tuple[float, bool]:
        y = math.tan(self.theta / 2.0)
        if current > 0.0:
            root_i = math.sqrt(current)
            phase = math.atan(y / root_i)
            next_phase = phase + root_i * self.dt
            spiked = next_phase >= math.pi / 2.0
            if math.isclose(math.cos(next_phase), 0.0, rel_tol=0.0, abs_tol=1e-15):
                return -math.pi, spiked
            return self._wrap_phase(2.0 * math.atan(root_i * math.tan(next_phase))), spiked
        if current == 0.0:
            denominator = 1.0 - y * self.dt
            if math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1e-15):
                return -math.pi, True
            next_y = y / denominator
            return self._wrap_phase(2.0 * math.atan(next_y)), denominator <= 0.0

        root_i = math.sqrt(-current)
        if math.isclose(y, -root_i, rel_tol=0.0, abs_tol=1e-15):
            return self.theta, False
        numerator_ratio = (y - root_i) / (y + root_i)
        try:
            evolved_ratio = numerator_ratio * math.exp(2.0 * root_i * self.dt)
        except OverflowError:
            return math.nan, False
        denominator = 1.0 - evolved_ratio
        spiked = numerator_ratio < 1.0 <= evolved_ratio or math.isclose(
            denominator,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        if spiked and math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1e-15):
            return -math.pi, True
        next_y = root_i * (1.0 + evolved_ratio) / denominator
        return self._wrap_phase(2.0 * math.atan(next_y)), spiked

    def _validate_event_packet_resolution(self, current: float) -> None:
        if current > 0.0 and math.sqrt(current) * self.dt > math.pi:
            raise ValueError("theta step can contain more than one source event")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        self._validate_event_packet_resolution(current)

        next_theta, spiked = self._exact_candidate(current)
        if not math.isfinite(next_theta):
            raise ValueError("exact-flow candidate must be finite")
        self.theta = self._wrap_phase(next_theta)
        return int(spiked)

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Return post-step phase plus aggregate event count through one backend.

        Parameters
        ----------
        n_steps:
            Number of updates (non-negative).
        current:
            Constant injected current (finite).
        backend:
            One of ``"auto"``, ``"python"``, ``"rust"``, ``"julia"``,
            ``"go"``, or ``"mojo"``. Auto probes Go, Julia, Mojo,
            compatible Rust, then Python, avoiding Julia initialisation when
            the Go shared library is available.
        """
        phase, events = self.simulate_complete(n_steps, current, backend)
        return phase, int(np.sum(events, dtype=np.int64))

    def simulate_complete(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        """Return aligned phase/events and atomically commit the final phase."""
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()
        self._validate_event_packet_resolution(current)

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
                    "Rust Theta backend requested but sc_neurocore_engine is unavailable."
                )
            packet = _backends.simulate_rust_complete(self.theta, self.dt, n_steps, current)
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia Theta backend requested but juliacall or the module is unavailable."
                )
            packet = _backends.simulate_julia_complete(self.theta, self.dt, n_steps, current)
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go Theta backend requested but libtheta.so is not built; run "
                    "go build -buildmode=c-shared -o libtheta.so theta.go in "
                    "accel/go/neurons/theta."
                )
            packet = _backends.simulate_go_complete(self.theta, self.dt, n_steps, current)
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo Theta backend requested but libtheta.so is not built; run "
                    "mojo build --emit shared-lib -o libtheta.so theta.mojo in "
                    "accel/mojo/kernels."
                )
            packet = _backends.simulate_mojo_complete(self.theta, self.dt, n_steps, current)
        else:
            packet = self._simulate_python_complete(n_steps, current)
        phase, events, final_theta = self._validated_complete_packet(packet, n_steps)
        self.theta = final_theta
        return phase, events

    def _simulate_python_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, float]:
        candidate = copy(self)
        phase = np.empty(n_steps, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        for t in range(n_steps):
            events[t] = candidate.step(current)
            phase[t] = candidate.theta
        return phase, events, candidate.theta

    @staticmethod
    def _validated_complete_packet(
        packet: tuple[object, object, float], n_steps: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], float]:
        raw_phase, raw_events, raw_final_theta = packet
        phase = np.ascontiguousarray(np.asarray(raw_phase, dtype=np.float64))
        event_values = np.asarray(raw_events)
        final_theta = float(raw_final_theta)
        if phase.shape != (n_steps,) or event_values.shape != (n_steps,):
            raise RuntimeError("Theta backend returned an invalid packet shape")
        if not np.all(np.isfinite(phase)) or not math.isfinite(final_theta):
            raise FloatingPointError("Theta backend returned non-finite phase")
        if not np.all((event_values == 0) | (event_values == 1)):
            raise RuntimeError("Theta backend returned non-binary events")
        events = np.ascontiguousarray(event_values, dtype=np.uint8)
        if (
            np.any(phase < -math.pi)
            or np.any(phase >= math.pi)
            or final_theta < -math.pi
            or final_theta >= math.pi
        ):
            raise RuntimeError("Theta backend returned phase outside [-pi, pi)")
        if n_steps and final_theta != float(phase[-1]):
            raise RuntimeError("Theta backend final state disagrees with its trace")
        return phase, events, final_theta

    def reset(self) -> None:
        """Restore the runtime phase while preserving the integration step."""
        self.theta = 0.0

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.theta):
            raise ValueError("runtime phase state must be finite")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("runtime dt must be finite and positive")
