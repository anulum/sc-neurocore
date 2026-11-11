// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Learnable Neuron Model

//! Learnable parameterized neuron dynamics.

#![deny(missing_docs)]

/// Learnable Neuron Model (LNM) — parameterised activation + decay.
#[derive(Clone, Debug)]
pub struct LearnableNeuronModel {
    /// Membrane state (dimensionless); reset to 0 by a spike.
    pub v: f64,
    /// Trainable membrane retention (leak) coefficient (dimensionless).
    pub alpha: f64,
    /// Trainable input weight (dimensionless).
    pub beta: f64,
    /// Trainable activation-nonlinearity weight (dimensionless).
    pub gamma: f64,
    /// Firing threshold (normalised) (dimensionless).
    pub v_threshold: f64,
    /// Slope of the learnable activation function (dimensionless).
    pub f_slope: f64,
    /// Horizontal shift of the learnable activation function (dimensionless).
    pub f_shift: f64,
}

impl LearnableNeuronModel {
    /// Create a neuron with the default parameters and initial state of its model descriptor.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            alpha: 0.9,
            beta: 0.1,
            gamma: 0.05,
            v_threshold: 1.0,
            f_slope: 5.0,
            f_shift: 0.5,
        }
    }
    /// Advance one step with the input `current` and return 1 on a spike, else 0.
    ///
    /// `v` becomes `alpha * v + beta * current + gamma * f(v)` with the logistic
    /// `f(v) = 1 / (1 + exp(-f_slope * (v - f_shift)))` of the previous `v`; reaching
    /// `v_threshold` resets `v` to 0. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let f_v = 1.0 / (1.0 + (-(self.f_slope * (self.v - self.f_shift))).exp());
        self.v = self.alpha * self.v + self.beta * current + self.gamma * f_v;
        if self.v >= self.v_threshold {
            self.v = 0.0;
            1
        } else {
            0
        }
    }
    /// Clear the membrane state; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}
impl Default for LearnableNeuronModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = LearnableNeuronModel::default();
        let constructed = LearnableNeuronModel::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn lnm_fires() {
        let mut n = LearnableNeuronModel::new();
        let t: i32 = (0..50).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn lnm_reset_clears_state() {
        let mut n = LearnableNeuronModel::new();
        for _ in 0..50 {
            n.step(2.0);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }

    #[test]
    fn lnm_bounded() {
        let mut n = LearnableNeuronModel::new();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn lnm_nan_no_panic() {
        LearnableNeuronModel::new().step(f64::NAN);
    }
}
