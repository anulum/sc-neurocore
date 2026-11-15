// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — KLIF Neuron

#![deny(missing_docs)]

/// KLIF — learnable LIF with scaling factor k. Eshraghian et al. 2021.
#[derive(Clone, Debug)]
pub struct KLIFNeuron {
    /// Membrane potential (dimensionless).
    pub v: f64,
    /// Learnable input gain; the update adds k times the input current (dimensionless).
    pub k: f64,
    /// Per-step leak factor, `exp(-dt / tau)`.
    pub alpha: f64,
    /// Firing threshold (normalised) (dimensionless).
    pub v_threshold: f64,
    /// Membrane potential reset to after a spike (normalised) (dimensionless).
    pub v_reset: f64,
}

impl KLIFNeuron {
    /// Create a neuron at rest with leak factor `exp(-dt / tau)` and input gain `k`; the
    /// threshold is 1 and the reset value 0.
    pub fn new(tau: f64, k: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            k,
            alpha: (-dt / tau).exp(),
            v_threshold: 1.0,
            v_reset: 0.0,
        }
    }

    /// Advance one step: `v` becomes `alpha * v + k * current`, and reaching `v_threshold`
    /// sets `v` to `v_reset` and returns 1, else 0. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        self.v = self.alpha * self.v + self.k * current;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    /// Set the membrane potential to 0 (not `v_reset`); the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for KLIFNeuron {
    fn default() -> Self {
        Self::new(10.0, 1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klif_fires() {
        let mut n = KLIFNeuron::default();
        let total: i32 = (0..50).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }
    #[test]
    fn klif_silent_without_input() {
        let mut n = KLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn klif_reset_clears_state() {
        let mut n = KLIFNeuron::default();
        for _ in 0..20 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn klif_bounded() {
        let mut n = KLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn klif_nan_no_panic() {
        KLIFNeuron::default().step(f64::NAN);
    }
}
