# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Biological and homeostatic plasticity adapters

"""Biological STDP, BCM, and homeostatic plasticity adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bioware_validation import (
    require_finite,
    require_nonnegative,
    require_nonnegative_int,
    require_positive,
)


@dataclass
class BiologicalSTDP:
    """Spike-Timing-Dependent Plasticity adapter for bio-hybrid loops.

    Bridges biological STDP time constants (∼20 ms) to SC clock
    rates (MHz) via a time-scaling factor. Computes ΔW from
    pre/post spike timing in biological time, then converts to
    Q8.8 weight updates for the SC domain.
    """

    tau_plus_ms: float = 20.0  # potentiation time constant
    tau_minus_ms: float = 20.0  # depression time constant
    a_plus: float = 0.01  # potentiation amplitude
    a_minus: float = 0.012  # depression amplitude (slightly > a_plus)
    w_max_q88: int = 512  # Q8.8 = 2.0
    w_min_q88: int = 0

    def __post_init__(self) -> None:
        """Validate time constants, amplitudes, and Q8.8 bounds."""
        require_positive(self.tau_plus_ms, "tau_plus_ms")
        require_positive(self.tau_minus_ms, "tau_minus_ms")
        require_nonnegative(self.a_plus, "a_plus")
        require_nonnegative(self.a_minus, "a_minus")
        require_nonnegative_int(self.w_min_q88, "w_min_q88")
        require_nonnegative_int(self.w_max_q88, "w_max_q88")
        if self.w_max_q88 < self.w_min_q88:
            raise ValueError("w_max_q88 must be >= w_min_q88")

    def compute_dw(self, dt_ms: float) -> float:
        """Compute weight change from spike timing difference.

        dt_ms = t_post - t_pre (positive = potentiation, negative = depression)
        """
        require_finite(dt_ms, "dt_ms")
        if dt_ms > 0:
            return float(self.a_plus * np.exp(-dt_ms / self.tau_plus_ms))
        elif dt_ms < 0:
            return float(-self.a_minus * np.exp(dt_ms / self.tau_minus_ms))
        return 0.0

    def update_weight(self, current_q88: int, dt_ms: float) -> int:
        """Update Q8.8 weight from spike timing."""
        require_nonnegative_int(current_q88, "current_q88")
        if not self.w_min_q88 <= current_q88 <= self.w_max_q88:
            raise ValueError("current_q88 must be inside configured weight bounds")
        dw = self.compute_dw(dt_ms)
        dw_q88 = int(dw * 256)  # Convert to Q8.8
        new_w = current_q88 + dw_q88
        return max(self.w_min_q88, min(self.w_max_q88, new_w))


@dataclass
class BCMPlasticity:
    """Bienenstock-Cooper-Munro plasticity adapter.

    Implements sliding-threshold BCM rule where the modification
    threshold θ tracks the postsynaptic firing rate. Converts
    biological firing rates to Q8.8 weight deltas.
    """

    tau_theta_ms: float = 1000.0  # threshold adaptation time constant
    learning_rate: float = 0.001
    theta: float = 0.0  # sliding threshold (internal state)
    w_max_q88: int = 512
    w_min_q88: int = 0

    def __post_init__(self) -> None:
        """Validate BCM dynamics and Q8.8 weight bounds."""
        require_positive(self.tau_theta_ms, "tau_theta_ms")
        require_nonnegative(self.learning_rate, "learning_rate")
        require_nonnegative(self.theta, "theta")
        require_nonnegative_int(self.w_min_q88, "w_min_q88")
        require_nonnegative_int(self.w_max_q88, "w_max_q88")
        if self.w_max_q88 < self.w_min_q88:
            raise ValueError("w_max_q88 must be >= w_min_q88")

    def update_theta(self, post_rate_hz: float, dt_ms: float) -> float:
        """Update the sliding threshold from postsynaptic activity."""
        require_nonnegative(post_rate_hz, "post_rate_hz")
        require_nonnegative(dt_ms, "dt_ms")
        alpha = dt_ms / self.tau_theta_ms
        target = post_rate_hz**2
        self.theta += alpha * (target - self.theta)
        return self.theta

    def compute_dw(self, pre_rate_hz: float, post_rate_hz: float) -> float:
        """BCM weight change: ΔW = η * x * y * (y - θ)."""
        require_nonnegative(pre_rate_hz, "pre_rate_hz")
        require_nonnegative(post_rate_hz, "post_rate_hz")
        require_nonnegative(self.theta, "theta")
        return self.learning_rate * pre_rate_hz * post_rate_hz * (post_rate_hz - self.theta)

    def update_weight(self, current_q88: int, pre_rate: float, post_rate: float) -> int:
        """Apply the BCM update to a saturated Q8.8 synaptic weight.

        Parameters
        ----------
        current_q88:
            Current synaptic weight encoded as Q8.8.
        pre_rate:
            Presynaptic firing rate in hertz.
        post_rate:
            Postsynaptic firing rate in hertz.

        Returns
        -------
        int
            Updated Q8.8 weight clamped to ``[w_min_q88, w_max_q88]``.
        """
        require_nonnegative_int(current_q88, "current_q88")
        if not self.w_min_q88 <= current_q88 <= self.w_max_q88:
            raise ValueError("current_q88 must be inside configured weight bounds")
        dw = self.compute_dw(pre_rate, post_rate)
        dw_q88 = int(dw * 256)
        new_w = current_q88 + dw_q88
        return max(self.w_min_q88, min(self.w_max_q88, new_w))


@dataclass
class HomeostaticPlasticity:
    """Intrinsic excitability scaling to maintain target firing rate.

    Implements homeostatic plasticity: if a neuron fires too fast,
    reduce its excitability (threshold up); too slow, increase it.
    Operates on Q8.8 threshold values.
    """

    target_rate_hz: float = 10.0
    tau_homeo_ms: float = 10000.0  # slow timescale (seconds)
    max_threshold_q88: int = 512  # Q8.8 = 2.0
    min_threshold_q88: int = 64  # Q8.8 = 0.25

    def __post_init__(self) -> None:
        """Validate target dynamics and Q8.8 threshold bounds."""
        require_nonnegative(self.target_rate_hz, "target_rate_hz")
        require_positive(self.tau_homeo_ms, "tau_homeo_ms")
        require_nonnegative_int(self.min_threshold_q88, "min_threshold_q88")
        require_nonnegative_int(self.max_threshold_q88, "max_threshold_q88")
        if self.max_threshold_q88 < self.min_threshold_q88:
            raise ValueError("max_threshold_q88 must be >= min_threshold_q88")

    def update_threshold(
        self,
        current_q88: int,
        observed_rate_hz: float,
        dt_ms: float,
    ) -> int:
        """Adjust threshold to drive firing rate toward target.

        Proportional homeostatic controller on a Q8.8 fixed-point
        threshold. ``alpha = dt_ms / tau_homeo_ms`` is the integration
        weight over the time step; the rate error (``observed − target``)
        is scaled by ``alpha·256`` so that a 1 Hz error integrated over
        one full time-constant shifts the threshold by 1.0 Q8.8 unit
        (i.e. by ``256`` in integer representation). Result clamped to
        ``[min_threshold_q88, max_threshold_q88]``.
        """
        require_nonnegative_int(current_q88, "current_q88")
        if not self.min_threshold_q88 <= current_q88 <= self.max_threshold_q88:
            raise ValueError("current_q88 must be inside configured threshold bounds")
        require_nonnegative(observed_rate_hz, "observed_rate_hz")
        require_nonnegative(dt_ms, "dt_ms")
        error = observed_rate_hz - self.target_rate_hz
        alpha = dt_ms / self.tau_homeo_ms
        delta_q88 = int(alpha * error * 256.0)
        new_q88 = current_q88 + delta_q88
        return max(self.min_threshold_q88, min(self.max_threshold_q88, new_q88))
