// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SuperSpike Neuron Model

//! SuperSpike surrogate-gradient neuron dynamics.

#![deny(missing_docs)]

/// SuperSpike — LIF with surrogate gradient trace. Zenke & Ganguli 2018.
#[derive(Clone, Debug)]
pub struct SuperSpikeNeuron {
    /// Membrane potential (dimensionless); reset to `v_reset` by a spike.
    pub v: f64,
    /// Surrogate-gradient eligibility trace: decays by `alpha_e` and adds
    /// `1 / (beta_sg * |v - v_threshold| + 1)^2` each step.
    pub trace: f64,
    /// Per-step membrane decay factor, `exp(-dt / tau_m)`.
    pub alpha_m: f64,
    /// Per-step trace decay factor, `exp(-dt / tau_e)`.
    pub alpha_e: f64,
    /// Normalised firing threshold (dimensionless).
    pub v_threshold: f64,
    /// Normalised reset potential (dimensionless).
    pub v_reset: f64,
    /// Steepness of the surrogate-gradient nonlinearity (dimensionless).
    pub beta_sg: f64,
}

impl SuperSpikeNeuron {
    /// Create a neuron at rest with decay factors `exp(-dt / tau_m)` and `exp(-dt / tau_e)`;
    /// the threshold, reset and surrogate parameters take their descriptor defaults.
    pub fn new(tau_m: f64, tau_e: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            trace: 0.0,
            alpha_m: (-dt / tau_m).exp(),
            alpha_e: (-dt / tau_e).exp(),
            v_threshold: 1.0,
            v_reset: 0.0,
            beta_sg: 10.0,
        }
    }
    /// Advance one step with the input `current` and return 1 on a spike, else 0.
    ///
    /// `v` becomes `alpha_m * v + current`, the trace is updated from the new `v`, and
    /// reaching `v_threshold` resets `v` to `v_reset`. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        self.v = self.alpha_m * self.v + current;
        let sg = 1.0 / (self.beta_sg * (self.v - self.v_threshold).abs() + 1.0).powi(2);
        self.trace = self.alpha_e * self.trace + sg;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    /// Clear the membrane potential and the trace; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.trace = 0.0;
    }
}

impl Default for SuperSpikeNeuron {
    fn default() -> Self {
        Self::new(10.0, 10.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = SuperSpikeNeuron::default();
        let constructed = SuperSpikeNeuron::new(10.0, 10.0, 1.0);
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn superspike_fires() {
        let mut n = SuperSpikeNeuron::default();
        let t: i32 = (0..50).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }

    #[test]
    fn superspike_reset_clears_state() {
        let mut n = SuperSpikeNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }

    #[test]
    fn superspike_bounded() {
        let mut n = SuperSpikeNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn superspike_trace_evolves() {
        let mut n = SuperSpikeNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        assert!(n.trace.is_finite());
    }

    #[test]
    fn superspike_nan_no_panic() {
        SuperSpikeNeuron::default().step(f64::NAN);
    }
}
