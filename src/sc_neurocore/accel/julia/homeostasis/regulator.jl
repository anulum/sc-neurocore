# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for homeostasis/regulator

module RegulatorAccel

using Statistics, LinearAlgebra

mutable struct SleepConsolidationState
    mean_firing_rate::Float64
    rate_variance::Float64
    ei_ratio::Float64
    weight_norm::Float64
    is_stable::Float64
    adjustments_made::Float64
    target_rate::Float64
    rate_tolerance::Float64
    threshold_step::Float64
    lr_scale_factor::Float64
    decay_exponent::Float64
    noise_amplitude::Float64
    duration_fraction::Float64
end

function SleepConsolidationState()
    SleepConsolidationState(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::SleepConsolidationState)
    status = "STABLE" if s.is_stable else "UNSTABLE"
    lines = [
        f"Network Stability: {status}",
        f"  Mean firing rate: {s.mean_firing_rate:.4f}",
        f"  Rate variance: {s.rate_variance:.4f}",
        f"  E/I ratio: {s.ei_ratio:.2f}",
        f"  Weight norm: {s.weight_norm:.4f}",
    ]
    if s.adjustments_made:  # pragma: no cover
        lines = push!(, f"  Adjustments: {', '.join(s.adjustments_made)}")
    return "\n".join(lines)
end

function regulate(s::SleepConsolidationState)
    self,
    firing_rates: np.ndarray,
    thresholds: np.ndarray,
    learning_rate: float,
    weights: list[np.ndarray] | nothing = nothing,
    ) -> tuple[np.ndarray, float, StabilityMetrics]
    mean_rate = float(firing_rates.mean())
    rate_var = float(firing_rates.var())
    metrics = StabilityMetrics(
        mean_firing_rate=mean_rate,
        rate_variance=rate_var,
    )
    if weights
        metrics.weight_norm = float(mean([norm(w) for w in weights]))
    new_thresholds = thresholds.copy()
    new_lr = learning_rate
    lo = s.target_rate * (1 - s.rate_tolerance)
    hi = s.target_rate * (1 + s.rate_tolerance)
    # Too active → raise thresholds
    if mean_rate > hi
        new_thresholds += s.threshold_step
        metrics.adjustments_made = push!(, f"thresholds +{s.threshold_step:.3f}")
        metrics.is_stable = false
    # Too quiet → lower thresholds
    elseif mean_rate < lo
        new_thresholds -= s.threshold_step
        metrics.adjustments_made = push!(, f"thresholds -{s.threshold_step:.3f}")
        metrics.is_stable = false
    # High variance → reduce LR
    if rate_var > s.target_rate * 2
        new_lr *= s.lr_scale_factor
        metrics.adjustments_made = push!(, f"lr *{s.lr_scale_factor}")
    return new_thresholds, new_lr, metrics
end

function apply(s::SleepConsolidationState)
    self,
    weights: list[np.ndarray],
    seed: int = 42,
    ) -> list[np.ndarray]
    rng = np.random.RandomState(seed)
    consolidated = []
    for w in weights
        abs_w = abs(w)
        # Power-law decay: larger weights decay more
        max_w = max(abs_w.max(), 1e-8)
        relative = abs_w / max_w
        decay_factor = 1.0 - s.duration_fraction * (relative^s.decay_exponent)
        decay_factor = clamp(decay_factor, 0.5, 1.0)
        # Apply decay
        w_new = w * decay_factor
        # Add spontaneous replay noise
        w_new += rng.randn(*w.shape) * s.noise_amplitude
        consolidated = push!(, w_new)
    return consolidated
end

function should_sleep(s::SleepConsolidationState, epoch, total_epochs)
    interval = max(1, int(1.0 / s.duration_fraction))
    return epoch > 0 && epoch % interval == 0
end

end # module RegulatorAccel
