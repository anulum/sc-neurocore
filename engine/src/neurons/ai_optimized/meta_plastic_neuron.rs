// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Meta-plastic neuron model

#![deny(missing_docs)]

/// Neuron with self-regulating meta-learning rate.
/// error_trace adapts learning speed: high error → learn faster, low error → stabilize.
#[derive(Clone, Debug)]
pub struct MetaPlasticNeuron {
    /// Membrane potential (dimensionless); a spike sets it to 0.
    pub v: f64,
    /// Low-pass trace of the reward error `|reward - expected_reward|`, with `tau_meta`.
    pub error_trace: f64,
    /// Running estimate of the reward, moved by `update_meta` at the rate `meta_lr`.
    pub expected_reward: f64,
    /// Membrane time constant (ms).
    pub tau: f64,
    /// Meta-learning error-trace time constant (ms).
    pub tau_meta: f64,
    /// Firing threshold (dimensionless).
    pub theta: f64,
    /// Baseline learning rate (dimensionless).
    pub lr0: f64,
    /// Sigmoid steepness of the meta-learning-rate modulation (dimensionless).
    pub kappa: f64,
    /// Target error level for the meta-learning controller (dimensionless).
    pub target_error: f64,
    /// Step (ms), shared by `step` and `update_meta`.
    pub dt: f64,
}

impl MetaPlasticNeuron {
    /// Create a neuron at rest with no error trace or reward estimate, the default
    /// parameters and `dt` = 1 ms.
    pub fn new() -> Self {
        Self {
            v: 0.0,
            error_trace: 0.0,
            expected_reward: 0.0,
            tau: 10.0,
            tau_meta: 500.0,
            theta: 1.0,
            lr0: 0.01,
            kappa: 5.0,
            target_error: 0.3,
            dt: 1.0,
        }
    }

    /// Advance one forward-Euler step of `v` towards `current` with `tau`, and return 1 when
    /// `v` reaches `theta`, which sets `v` to 0; else 0. The learning state is untouched.
    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-self.v + current) / self.tau * self.dt;
        if self.v >= self.theta {
            self.v = 0.0;
            1
        } else {
            0
        }
    }

    /// Learn from `reward`. The error trace takes a step towards
    /// `|reward - expected_reward|` with `tau_meta`. Then `expected_reward` moves towards
    /// `reward` by the fraction `meta_lr()`, evaluated with the new trace.
    pub fn update_meta(&mut self, reward: f64) {
        let error = (reward - self.expected_reward).abs();
        self.error_trace += (-self.error_trace + error) / self.tau_meta * self.dt;
        let meta_lr = self.meta_lr();
        self.expected_reward += meta_lr * (reward - self.expected_reward);
    }

    /// Return the learning rate `lr0 / (1 + exp(-kappa * (error_trace - target_error)))`.
    pub fn meta_lr(&self) -> f64 {
        self.lr0 / (1.0 + (-self.kappa * (self.error_trace - self.target_error)).exp())
    }

    /// Clear `v`, the error trace and the reward estimate; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.error_trace = 0.0;
        self.expected_reward = 0.0;
    }
}

impl Default for MetaPlasticNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_plastic_fires() {
        let mut n = MetaPlasticNeuron::new();
        let total: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(total > 0);
    }

    #[test]
    fn meta_plastic_adapts_lr() {
        let mut n = MetaPlasticNeuron::new();
        let lr_before = n.meta_lr();
        for _ in 0..100 {
            n.step(2.0);
            n.update_meta(1.0);
        }
        let lr_after = n.meta_lr();
        assert!((lr_after - lr_before).abs() > 1e-6);
    }

    #[test]
    fn meta_plastic_reset() {
        let mut n = MetaPlasticNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
            n.update_meta(1.0);
        }
        n.reset();
        assert_eq!(n.v, 0.0);
        assert_eq!(n.error_trace, 0.0);
        assert_eq!(n.expected_reward, 0.0);
    }
}
