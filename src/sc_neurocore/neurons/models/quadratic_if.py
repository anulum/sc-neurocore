# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Latham QIF + preserved symmetric SC profile

"""Source-explicit exact-flow Quadratic IF dynamics and SC compatibility."""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel import quadratic_if as _backends

QuadraticIFProfile = Literal["sc_symmetric", "latham_2000"]


@dataclass
class QuadraticIFNeuron:
    """Quadratic IF with Latham-source and preserved SC profiles.

    dv/dt = v² + I
    Reset when v >= v_peak.

    ``QuadraticIFNeuron()`` preserves the historical symmetric SC boundary.
    :meth:`latham_2000` constructs the count-bearing isolated scalar
    normalisation of Latham et al. equations (1), (2), and (5a):
    ``v=-1``, ``v_reset=-3``, ``v_peak=31/3``, and ``dt=0.05``.  Here ``+1`` is
    the unstable equilibrium, not the event apex.  The exact held-current
    Riccati map is a catalogue numerical specialisation of the source ODE.

    Reference: Latham, P.E. et al. (2000). J. Neurophysiol. 83:808–827.
    doi:10.1152/jn.2000.83.2.808.
    """

    v: float = -1.0
    v_reset: float = -1.0
    v_peak: float = 1.0
    dt: float = 0.01
    profile: QuadraticIFProfile = "sc_symmetric"

    def __post_init__(self) -> None:
        """Validate the finite ordered state and integration contract."""
        if self.profile not in ("sc_symmetric", "latham_2000"):
            raise ValueError(f"unsupported QuadraticIF profile: {self.profile!r}")
        for field in ("v", "v_reset", "v_peak"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.v >= self.v_peak:
            raise ValueError("v must be below v_peak")
        if self.v_reset >= self.v_peak:
            raise ValueError("v_peak must be greater than v_reset")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")

    @classmethod
    def latham_2000(
        cls,
        *,
        v: float = -1.0,
        v_reset: float = -3.0,
        v_peak: float = 31.0 / 3.0,
        dt: float = 0.05,
    ) -> QuadraticIFNeuron:
        """Construct Latham et al.'s normalized numerical source profile."""
        return cls(v=v, v_reset=v_reset, v_peak=v_peak, dt=dt, profile="latham_2000")

    @classmethod
    def sc_symmetric_compatibility(
        cls,
        *,
        v: float = -1.0,
        v_reset: float = -1.0,
        v_peak: float = 1.0,
        dt: float = 0.01,
    ) -> QuadraticIFNeuron:
        """Construct the preserved symmetric finite-boundary SC profile."""
        return cls(v=v, v_reset=v_reset, v_peak=v_peak, dt=dt, profile="sc_symmetric")

    def _exact_candidate(self, current: float) -> tuple[float, bool]:
        if current > 0.0:
            root_i = math.sqrt(current)
            phase = math.atan(self.v / root_i)
            peak_phase = math.atan(self.v_peak / root_i)
            next_phase = phase + root_i * self.dt
            if next_phase >= peak_phase or next_phase >= math.pi / 2.0:
                return self.v_reset, True
            return root_i * math.tan(next_phase), False
        if current == 0.0:
            denominator = 1.0 - self.v * self.dt
            if denominator <= 0.0:
                return self.v_reset, True
            next_v = self.v / denominator
            return (self.v_reset, True) if next_v >= self.v_peak else (next_v, False)

        root_i = math.sqrt(-current)
        if math.isclose(self.v, -root_i, rel_tol=0.0, abs_tol=1e-15):
            return self.v, False
        numerator_ratio = (self.v - root_i) / (self.v + root_i)
        try:
            evolved_ratio = numerator_ratio * math.exp(2.0 * root_i * self.dt)
        except OverflowError:
            return math.nan, False
        denominator = 1.0 - evolved_ratio
        if numerator_ratio < 1.0 <= evolved_ratio or math.isclose(
            denominator, 0.0, rel_tol=0.0, abs_tol=1e-15
        ):
            return self.v_reset, True
        next_v = root_i * (1.0 + evolved_ratio) / denominator
        return (self.v_reset, True) if next_v >= self.v_peak else (next_v, False)

    def step(self, current: float) -> int:
        """Advance one exact constant-current Riccati-flow update.

        Parameters
        ----------
        current:
            Finite constant drive over this update.

        Returns
        -------
        int
            One when the within-step flow reaches ``v_peak``; otherwise zero.
        """
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self.__post_init__()
        next_v, spiked = self._exact_candidate(current)
        if not math.isfinite(next_v):
            raise ValueError("exact-flow candidate must be finite")
        self.v = next_v
        return int(spiked)

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance a batch and return its post-step trace and event count.

        Parameters
        ----------
        n_steps:
            Non-negative number of sequential exact-flow updates.
        current:
            Constant injected current (finite).
        backend:
            One of ``"auto"``, ``"python"``, ``"rust"``, ``"julia"``,
            ``"go"``, or ``"mojo"``. Auto uses the committed production order
            Go, Julia, Mojo, compatible Rust, then Python; the Go shared library
            avoids Julia runtime initialisation on the first call.
        """
        voltage, events = self.simulate_complete(n_steps, current, backend)
        return voltage, int(np.sum(events, dtype=np.int64))

    def simulate_complete(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
        """Return aligned voltage/events and commit valid final state atomically."""
        if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self.__post_init__()

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
                    "Rust QuadraticIF backend requested but sc_neurocore_engine is unavailable."
                )
            packet = _backends.simulate_rust_complete(*self._native_arguments(n_steps, current))
        elif selected == "julia":
            if not _backends.ensure_julia_loaded():
                raise RuntimeError(
                    "Julia QuadraticIF backend requested but juliacall or the module is "
                    "unavailable."
                )
            packet = _backends.simulate_julia_complete(*self._native_arguments(n_steps, current))
        elif selected == "go":
            if not _backends.ensure_go_loaded():
                raise RuntimeError(
                    "Go QuadraticIF backend requested but libquadratic_if.so is not built; "
                    "run go build -buildmode=c-shared -o libquadratic_if.so "
                    "quadratic_if.go in accel/go/neurons/quadratic_if."
                )
            packet = _backends.simulate_go_complete(*self._native_arguments(n_steps, current))
        elif selected == "mojo":
            if not _backends.ensure_mojo_loaded():
                raise RuntimeError(
                    "Mojo QuadraticIF backend requested but libquadratic_if.so is not built; "
                    "run mojo build --emit shared-lib -o libquadratic_if.so "
                    "quadratic_if.mojo in accel/mojo/kernels."
                )
            packet = _backends.simulate_mojo_complete(*self._native_arguments(n_steps, current))
        else:
            packet = self._simulate_python_complete(n_steps, current)
        voltage, events, final_v = self._validated_complete_packet(packet, n_steps)
        self.v = final_v
        return voltage, events

    def _native_arguments(
        self, n_steps: int, current: float
    ) -> tuple[float, float, float, float, bool, int, float]:
        return (
            self.v,
            self.v_reset,
            self.v_peak,
            self.dt,
            self.profile == "latham_2000",
            n_steps,
            current,
        )

    def _simulate_python_complete(
        self, n_steps: int, current: float
    ) -> tuple[object, object, float]:
        candidate = copy(self)
        voltage = np.empty(n_steps, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        for t in range(n_steps):
            events[t] = candidate.step(current)
            voltage[t] = candidate.v
        return voltage, events, candidate.v

    @staticmethod
    def _validated_complete_packet(
        packet: tuple[object, object, float], n_steps: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], float]:
        raw_voltage, raw_events, raw_final_v = packet
        voltage = np.ascontiguousarray(np.asarray(raw_voltage, dtype=np.float64))
        events = np.ascontiguousarray(np.asarray(raw_events, dtype=np.uint8))
        final_v = float(raw_final_v)
        if voltage.shape != (n_steps,) or events.shape != (n_steps,):
            raise RuntimeError("QuadraticIF backend returned an invalid packet shape")
        if not np.all(np.isfinite(voltage)) or not math.isfinite(final_v):
            raise FloatingPointError("QuadraticIF backend returned non-finite voltage")
        if np.any(events > 1):
            raise RuntimeError("QuadraticIF backend returned non-binary events")
        if n_steps and final_v != float(voltage[-1]):
            raise RuntimeError("QuadraticIF backend final state disagrees with its trace")
        return voltage, events, final_v

    def reset(self) -> None:
        """Restore the runtime voltage while preserving configured parameters."""
        self.v = self.v_reset


class SCSymmetricQuadraticIFNeuron(QuadraticIFNeuron):
    """Count-neutral explicit identity for the preserved symmetric SC profile."""

    def __init__(
        self,
        v: float = -1.0,
        v_reset: float = -1.0,
        v_peak: float = 1.0,
        dt: float = 0.01,
    ) -> None:
        super().__init__(v, v_reset, v_peak, dt, "sc_symmetric")
