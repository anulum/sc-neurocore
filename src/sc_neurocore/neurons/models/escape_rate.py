# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gerstner 2000 — stochastic threshold (escape noise model)

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons._stochastic_threshold import (
    DEFAULT_LFSR16_SEED,
    Lfsr16Threshold,
)
from sc_neurocore.utils.numerics import safe_exp


@dataclass
class EscapeRateNeuron:
    """Gerstner 2000 — stochastic threshold (escape noise model).

    Membrane dynamics use the exact constant-current RC flow before evaluating
    the finite-step escape hazard.

    Reference: Gerstner, W. (2000). Neural Comput. 12:43–89.
    """

    v: float = -70.0
    v_rest: float = -70.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    rho_0: float = 0.001
    delta_u: float = 3.0
    resistance: float = 1.0
    dt: float = 1.0
    seed: int | None = DEFAULT_LFSR16_SEED

    def __post_init__(self) -> None:
        self._validate_runtime_state()
        self._rng = Lfsr16Threshold(self.seed)

    @property
    def initial_seed(self) -> int:
        """Return the concrete seed used by this instance."""
        return self._rng.initial_seed

    @property
    def rng_state(self) -> int:
        """Return the current canonical LFSR16 state."""
        return self._rng.state

    def _validate_runtime_state(self) -> None:
        for field in ("v", "v_rest", "v_reset", "v_threshold"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        for field in ("tau_m", "rho_0", "delta_u", "resistance", "dt"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")

    def _spike_probability(self, voltage: float) -> float:
        if not math.isfinite(voltage):
            raise ValueError("voltage candidate must be finite")
        rate = self.rho_0 * safe_exp((voltage - self.v_threshold) / self.delta_u)
        hazard = rate * self.dt
        if not math.isfinite(hazard) or hazard < 0.0:
            raise ValueError("escape hazard must be finite and non-negative")
        probability = -math.expm1(-hazard)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("spike probability must remain finite and bounded")
        return probability

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        voltage = self._exact_voltage_candidate(current)
        p_spike = self._spike_probability(voltage)
        if self._rng.trial(p_spike):
            self.v = self.v_reset
            return 1
        self.v = voltage
        return 0

    def simulate(
        self,
        n_steps: int,
        current: float = 0.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Run a constant-current trace on Python or one real native backend.

        Every backend receives the complete physical state and the live LFSR
        state. A successful batch atomically commits its final voltage and RNG
        state; unavailable or rejected native runs leave the instance unchanged.
        """
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        current = float(current)
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_runtime_state()

        from sc_neurocore.accel import escape_rate as backends

        selected = backend
        if selected == "auto":
            if backends._HAS_RUST:
                selected = "rust"
            elif backends.ensure_mojo_loaded():
                selected = "mojo"
            elif backends.ensure_go_loaded():
                selected = "go"
            elif backends.ensure_julia_loaded():
                selected = "julia"
            else:
                selected = "python"

        previous_v = self.v
        previous_rng = self.rng_state
        try:
            if selected == "python":
                result = self._simulate_python(n_steps, current)
            else:
                loader = {
                    "rust": lambda: backends._HAS_RUST,
                    "julia": backends.ensure_julia_loaded,
                    "go": backends.ensure_go_loaded,
                    "mojo": backends.ensure_mojo_loaded,
                }[selected]
                if not loader():
                    raise RuntimeError(f"{selected.title()} EscapeRate backend is unavailable.")
                runner = {
                    "rust": backends.simulate_rust,
                    "julia": backends.simulate_julia,
                    "go": backends.simulate_go,
                    "mojo": backends.simulate_mojo,
                }[selected]
                result = runner(
                    self.v,
                    self.v_rest,
                    self.v_reset,
                    self.v_threshold,
                    self.tau_m,
                    self.rho_0,
                    self.delta_u,
                    self.resistance,
                    self.dt,
                    self.rng_state,
                    n_steps,
                    current,
                )
            trace, events, final_v, final_rng = backends._normalise_result(*result)
            trace_array = np.ascontiguousarray(trace, dtype=np.float64)
            spike_count = int(np.sum(events, dtype=np.int64))
            self._rng.restore(final_rng)
            self.v = final_v
        except Exception:
            self.v = previous_v
            self._rng.restore(previous_rng)
            raise

        return trace_array, spike_count

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], float, int]:
        """Execute the canonical Python recurrence without a surrogate path."""
        trace = np.empty(n_steps, dtype=np.float64)
        events = np.empty(n_steps, dtype=np.uint8)
        for index in range(n_steps):
            events[index] = self.step(current)
            trace[index] = self.v
        return trace, events, self.v, self.rng_state

    def reset(self) -> None:
        self.v = self.v_rest
        self._rng.reset()

    def _exact_voltage_candidate(self, current: float) -> float:
        steady_state = self.v_rest + self.resistance * current
        decay = math.exp(-self.dt / self.tau_m)
        voltage = steady_state + (self.v - steady_state) * decay
        if (
            not math.isfinite(steady_state)
            or not math.isfinite(decay)
            or not math.isfinite(voltage)
        ):
            raise ValueError("voltage candidate must be finite")
        return voltage
