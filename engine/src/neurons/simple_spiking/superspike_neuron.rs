// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SuperSpike Neuron Model

//! SuperSpike surrogate-gradient neuron dynamics.

/// SuperSpike — LIF with surrogate gradient trace. Zenke & Ganguli 2018.
#[derive(Clone, Debug)]
pub struct SuperSpikeNeuron {
    pub v: f64,
    pub trace: f64,
    pub alpha_m: f64,
    pub alpha_e: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub beta_sg: f64,
}

impl SuperSpikeNeuron {
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
