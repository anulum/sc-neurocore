# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Poisson spike generator — stochastic firing at rate λ

"""Generate reproducible homogeneous Poisson binary event streams."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons._stochastic_threshold import DEFAULT_LFSR16_SEED, Lfsr16Threshold


@dataclass
class PoissonNeuron:
    """Generate homogeneous Poisson events in discrete binary time bins.

    Parameters
    ----------
    rate_hz : float, default=100.0
        Non-negative homogeneous event rate in hertz.
    dt_ms : float, default=1.0
        Positive bin width in milliseconds. Multiple arrivals within one bin
        collapse to one event.
    seed : int or None, default=0xACE1
        Replay seed for the canonical 16-bit LFSR. ``None`` draws one concrete
        non-zero seed from system entropy and retains it for subsequent resets.

    Notes
    -----
    Each accepted bin uses the exact finite-interval event probability
    ``1 - exp(-rate_hz * dt_ms / 1000)`` and advances the shared LFSR16 by
    exactly one trial (eight primitive shifts). The generator therefore models
    a binary-bin observation of a homogeneous Poisson process, not an
    unbounded within-bin arrival count.

    References
    ----------
    Gerstner, W., Kistler, W. M., Naud, R., & Paninski, L. (2014).
    *Neuronal Dynamics*, Sections 7.2 and 7.7.
    https://doi.org/10.1017/CBO9781107447615
    """

    rate_hz: float = 100.0
    dt_ms: float = 1.0
    seed: int | None = DEFAULT_LFSR16_SEED
    _rng: Lfsr16Threshold = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate physical parameters and initialise the replayable RNG."""
        self._validate_runtime_state()
        self._probability(self.rate_hz)
        self._rng = Lfsr16Threshold(self.seed)

    @property
    def initial_seed(self) -> int:
        """Return the concrete seed restored by :meth:`reset`.

        Returns
        -------
        int
            Non-zero 16-bit replay seed. For ``seed=None`` this is the concrete
            entropy-derived value chosen during construction.
        """
        return self._rng.initial_seed

    @property
    def rng_state(self) -> int:
        """Return the live canonical LFSR16 state.

        Returns
        -------
        int
            Current non-zero 16-bit state after all accepted trials.
        """
        return self._rng.state

    def _validate_runtime_state(self) -> None:
        if not math.isfinite(self.rate_hz) or self.rate_hz < 0.0:
            raise ValueError("rate_hz must be finite and non-negative")
        if not math.isfinite(self.dt_ms) or self.dt_ms <= 0.0:
            raise ValueError("dt_ms must be finite and positive")

    def _probability(self, rate_hz: float) -> float:
        if not math.isfinite(rate_hz) or rate_hz < 0.0:
            raise ValueError("rate_hz must be finite and non-negative")
        hazard = rate_hz * self.dt_ms / 1000.0
        if not math.isfinite(hazard) or hazard < 0.0:
            raise ValueError("interval hazard must be finite and non-negative")
        probability = -math.expm1(-hazard)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("spike probability must remain finite and bounded")
        return probability

    def step(self, rate_override: float = -1.0) -> int:
        """Advance one binary time bin and return whether an event occurred.

        Parameters
        ----------
        rate_override : float, default=-1.0
            Rate in hertz for this bin. A negative value selects the configured
            :attr:`rate_hz`; zero or a positive value overrides it.

        Returns
        -------
        int
            ``1`` when an event occurs, otherwise ``0``.

        Raises
        ------
        ValueError
            If the configured state or supplied rate is non-finite or outside
            its physical domain.

        Notes
        -----
        A successful call advances the LFSR state exactly once. Validation
        failures occur before the stochastic trial and leave the state intact.
        """
        if not math.isfinite(rate_override):
            raise ValueError("rate_override must be finite")
        self._validate_runtime_state()
        r = self.rate_hz if rate_override < 0 else rate_override
        p = self._probability(r)
        return int(self._rng.trial(p))

    def simulate(
        self,
        n_steps: int,
        rate_override: float = -1.0,
        backend: str = "auto",
    ) -> tuple[npt.NDArray[np.uint8], int]:
        """Return the binary event trace from Python or one real native backend.

        Parameters
        ----------
        n_steps : int
            Number of binary time bins to generate.
        rate_override : float, default=-1.0
            Rate in hertz for every bin. A negative value selects
            :attr:`rate_hz`.
        backend : {"auto", "python", "rust", "julia", "go", "mojo"}, default="auto"
            Execution lane. ``auto`` follows the maintained production order:
            Rust, Mojo, Go, Julia, then Python.

        Returns
        -------
        events : numpy.ndarray
            Contiguous one-dimensional ``uint8`` array containing only zeroes
            and ones, with length ``n_steps``.
        event_count : int
            Sum of the returned binary event trace.

        Raises
        ------
        ValueError
            If a numeric input, step count, backend name, or configured state is
            invalid.
        RuntimeError
            If an explicitly requested native backend is unavailable.
        FloatingPointError
            If a native backend rejects the contract or returns malformed data.

        Notes
        -----
        Every lane receives the complete physical contract and current LFSR
        state. A successful batch atomically commits the returned RNG state;
        unavailable, malformed, or rejected native results leave the instance
        unchanged.
        """
        if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        if backend not in ("auto", "python", "rust", "julia", "go", "mojo"):
            raise ValueError(f"backend must be auto/python/rust/julia/go/mojo, got {backend!r}")
        rate_override = float(rate_override)
        if not math.isfinite(rate_override):
            raise ValueError("rate_override must be finite")
        self._validate_runtime_state()

        from sc_neurocore.accel import poisson as backends

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

        previous_rng = self.rng_state
        try:
            if selected == "python":
                result = self._simulate_python(n_steps, rate_override)
            else:
                loader = {
                    "rust": lambda: backends._HAS_RUST,
                    "julia": backends.ensure_julia_loaded,
                    "go": backends.ensure_go_loaded,
                    "mojo": backends.ensure_mojo_loaded,
                }[selected]
                if not loader():
                    raise RuntimeError(f"{selected.title()} Poisson backend is unavailable.")
                runner = {
                    "rust": backends.simulate_rust,
                    "julia": backends.simulate_julia,
                    "go": backends.simulate_go,
                    "mojo": backends.simulate_mojo,
                }[selected]
                result = runner(
                    self.rate_hz,
                    self.dt_ms,
                    self.rng_state,
                    n_steps,
                    rate_override,
                )
            events, final_rng = backends._normalise_result(*result)
            self._rng.restore(final_rng)
        except Exception:
            self._rng.restore(previous_rng)
            raise

        return events, int(np.sum(events, dtype=np.int64))

    def _simulate_python(
        self,
        n_steps: int,
        rate_override: float,
    ) -> tuple[npt.NDArray[np.uint8], int]:
        """Execute the canonical binary-bin recurrence without a surrogate path."""
        events = np.empty(n_steps, dtype=np.uint8)
        for index in range(n_steps):
            events[index] = self.step(rate_override)
        return events, self.rng_state

    def reset(self) -> None:
        """Restore the construction-time replay seed.

        Notes
        -----
        The configured rate and bin width are preserved. Subsequent calls replay
        the same event stream for a fixed execution contract.
        """
        self._rng.reset()
