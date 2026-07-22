// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive-threshold mixture-of-experts neuron model

/// Adaptive threshold spiking neuron matching the SpikingBrain architecture.
///
/// Converts activations into integer spike counts via data-dependent threshold,
/// enabling addition-based event-driven computation with ~69% sparsity.
///
/// Exact equations from arXiv:2509.05276v2 (SpikingBrain Technical Report):
///
///   V_th(x) = (1/k) · mean(|x|)          (adaptive threshold)
///   v[t+1] = v[t] - V_th · s[t] + x[t+1] (membrane with soft reset)
///   s_INT = round(v_T / V_th)              (integer spike count)
///
/// In time-collapsed mode: v_T = x, s_INT = round(x / V_th).
/// Parameter k controls the firing rate / sparsity trade-off.
///
/// Reference: SpikingBrain-1.0, arXiv:2509.05276v2, September 2025.
#[derive(Clone, Debug)]
pub struct AdaptiveThresholdMoENeuron {
    /// Membrane potential.
    pub v: f64,
    /// Current adaptive threshold.
    pub v_th: f64,
    /// Firing rate control parameter (higher k → lower threshold → more spikes).
    pub k: f64,
    /// Running EMA of |input| for threshold computation.
    mean_abs_x: f64,
    /// EMA decay for mean estimation.
    ema_alpha: f64,
}

impl AdaptiveThresholdMoENeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            v_th: 1.0,
            k: 4.0,
            mean_abs_x: 0.0,
            ema_alpha: 0.1,
        }
    }

    pub fn with_k(k: f64) -> Self {
        Self { k, ..Self::new() }
    }

    /// Returns integer spike count (0 or more) — not binary.
    ///
    /// Implements: V_th = (1/k)·mean(|x|), s = round(v/V_th), soft reset v -= V_th·s.
    pub fn step(&mut self, current: f64) -> i32 {
        // Update running mean of |activation|.
        self.mean_abs_x = (1.0 - self.ema_alpha) * self.mean_abs_x + self.ema_alpha * current.abs();

        // Adaptive threshold: V_th = (1/k) · mean(|x|).
        self.v_th = if self.mean_abs_x > 1e-12 {
            self.mean_abs_x / self.k
        } else {
            1.0 // fallback to avoid division by near-zero
        };

        // Membrane: v[t+1] = v[t] + x[t+1] (integrate input).
        self.v += current;

        // Integer spike count: s_INT = round(v / V_th).
        let s_int = if self.v_th > 1e-12 {
            (self.v / self.v_th).round() as i32
        } else {
            0
        };

        // Soft reset: v -= V_th · s.
        if s_int != 0 {
            self.v -= self.v_th * s_int as f64;
        }

        s_int.max(0) // non-negative spike counts
    }

    /// Time-collapsed single-step mode: s_INT = round(x / V_th).
    pub fn step_collapsed(&mut self, activation: f64) -> i32 {
        self.mean_abs_x =
            (1.0 - self.ema_alpha) * self.mean_abs_x + self.ema_alpha * activation.abs();
        self.v_th = if self.mean_abs_x > 1e-12 {
            self.mean_abs_x / self.k
        } else {
            1.0
        };
        let s_int = (activation / self.v_th).round() as i32;
        s_int.max(0)
    }

    /// Current activation sparsity estimate (1 if below threshold, 0 if firing).
    pub fn sparsity(&self) -> f64 {
        if self.v.abs() < self.v_th {
            1.0
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
        self.mean_abs_x = 0.0;
        self.v_th = 1.0;
    }
}

impl Default for AdaptiveThresholdMoENeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_threshold_fires_integer_counts() {
        let mut n = AdaptiveThresholdMoENeuron::new();
        let mut total_spikes = 0;
        for _ in 0..100 {
            total_spikes += n.step(2.0);
        }
        assert!(total_spikes > 0, "Must fire with positive input");
        // V_th adapts to mean(|x|)/k = 2.0/4.0 = 0.5, so round(2.0/0.5) = 4 per step.
        assert!(
            total_spikes > 100,
            "Should produce multi-spike counts, got {total_spikes}"
        );
    }

    #[test]
    fn adaptive_threshold_adapts_to_input_scale() {
        let mut n = AdaptiveThresholdMoENeuron::new();
        // Feed large inputs to set mean_abs_x.
        for _ in 0..50 {
            n.step(10.0);
        }
        let th_large = n.v_th;
        n.reset();
        // Feed small inputs.
        for _ in 0..50 {
            n.step(0.1);
        }
        let th_small = n.v_th;
        assert!(
            th_large > th_small,
            "Larger input → larger threshold: {th_large:.4} > {th_small:.4}"
        );
    }

    #[test]
    fn adaptive_threshold_collapsed_mode() {
        let mut n = AdaptiveThresholdMoENeuron::with_k(2.0);
        // Warm up threshold.
        for _ in 0..20 {
            n.step_collapsed(5.0);
        }
        let s = n.step_collapsed(5.0);
        // V_th ≈ 5.0/2.0 = 2.5, s ≈ round(5.0/2.5) = 2.
        assert!(s >= 1, "Collapsed mode must fire, got {s}");
    }

    #[test]
    fn adaptive_threshold_sparsity() {
        // Varying input with some near-zero values → sparse activations.
        let mut n = AdaptiveThresholdMoENeuron::with_k(4.0);
        let mut zeros = 0;
        let total = 200;
        for i in 0..total {
            // Alternate strong and near-zero input.
            let input = if i % 3 == 0 { 2.0 } else { 0.01 };
            if n.step(input) == 0 {
                zeros += 1;
            }
        }
        let sparsity = zeros as f64 / total as f64;
        assert!(
            sparsity > 0.1,
            "Should have some sparsity with varying input, got {sparsity:.2}"
        );
    }
}
