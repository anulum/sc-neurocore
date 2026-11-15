// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Closed-form Continuous-depth Neuron

#![deny(missing_docs)]

/// Closed-form Continuous-depth (CfC) neuron. Hasani et al. 2022.
#[derive(Clone, Debug)]
pub struct ClosedFormContinuousNeuron {
    /// State (dimensionless); a spike subtracts `v_threshold` from it.
    pub x: f64,
    /// Input weight modulating the input-dependent time constant (dimensionless).
    pub w_tau: f64,
    /// Recurrent (self) weight on the membrane state (dimensionless).
    pub w_x: f64,
    /// Input weight on the injected drive (dimensionless).
    pub w_in: f64,
    /// Baseline membrane time constant (ms).
    pub tau_base: f64,
    /// Constant bias term (dimensionless).
    pub bias: f64,
    /// Normalised firing threshold (dimensionless).
    pub v_threshold: f64,
    /// Step length (ms), in the time unit of `tau_base`.
    pub dt: f64,
}

impl ClosedFormContinuousNeuron {
    /// Create a neuron with the default parameters and initial state of its model descriptor.
    pub fn new() -> Self {
        Self {
            x: 0.0,
            w_tau: -0.5,
            w_x: 0.8,
            w_in: 1.0,
            tau_base: 10.0,
            bias: 0.0,
            v_threshold: 1.0,
            dt: 1.0,
        }
    }

    /// Advance one step with the input `current` and return 1 on a spike, else 0.
    ///
    /// `x` relaxes exactly over `dt` towards `tanh(w_x * x + w_in * current)` with the time
    /// constant `max(tau_base * sigmoid(w_tau * current + bias), 0.1)`. Reaching
    /// `v_threshold` subtracts it from `x` (soft reset). The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let sigma_tau = 1.0 / (1.0 + (-(self.w_tau * current + self.bias)).exp());
        let tau_eff = (self.tau_base * sigma_tau).max(0.1);
        let f_target = (self.w_x * self.x + self.w_in * current).tanh();
        let decay = (-self.dt / tau_eff).exp();
        self.x = self.x * decay + f_target * (1.0 - decay);
        if self.x >= self.v_threshold {
            self.x -= self.v_threshold;
            1
        } else {
            0
        }
    }

    /// Clear the state; the parameters are kept.
    pub fn reset(&mut self) {
        self.x = 0.0;
    }
}

impl Default for ClosedFormContinuousNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfc_activates() {
        let mut n = ClosedFormContinuousNeuron {
            v_threshold: 0.9,
            ..ClosedFormContinuousNeuron::new()
        };
        let total: i32 = (0..100).map(|_| n.step(5.0)).sum();
        assert!(total > 0);
    }
    #[test]
    fn cfc_reset_clears_state() {
        let mut n = ClosedFormContinuousNeuron::new();
        for _ in 0..50 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.x - 0.0).abs() < 1e-10);
    }
    #[test]
    fn cfc_bounded() {
        let mut n = ClosedFormContinuousNeuron {
            v_threshold: 0.9,
            ..ClosedFormContinuousNeuron::new()
        };
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.x.is_finite());
    }
    #[test]
    fn cfc_nan_no_panic() {
        ClosedFormContinuousNeuron::new().step(f64::NAN);
    }
}
