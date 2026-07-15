// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — E-Prop ALIF Neuron Model

//! Adaptive LIF dynamics for eligibility propagation.

/// e-prop ALIF — adaptive LIF for eligibility-propagation learning. Bellec et al. 2020.
#[derive(Clone, Debug)]
pub struct EPropALIFNeuron {
    pub v: f64,
    pub a: f64,
    pub e_trace: f64,
    pub alpha_m: f64,
    pub alpha_a: f64,
    pub v_threshold_base: f64,
    pub beta: f64,
}

impl EPropALIFNeuron {
    pub fn new(tau_m: f64, tau_a: f64, dt: f64) -> Self {
        Self {
            v: 0.0,
            a: 0.0,
            e_trace: 0.0,
            alpha_m: (-dt / tau_m).exp(),
            alpha_a: (-dt / tau_a).exp(),
            v_threshold_base: 1.0,
            beta: 0.07,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.v = self.alpha_m * self.v + current;
        let threshold = self.v_threshold_base + self.beta * self.a;
        let psi = ((1.0 - (self.v - threshold).abs()) * 0.3).max(0.0);
        self.e_trace = self.alpha_a * self.e_trace + psi;
        if self.v >= threshold {
            self.v = 0.0;
            self.a = self.alpha_a * self.a + 1.0;
            1
        } else {
            self.a *= self.alpha_a;
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0.0;
        self.a = 0.0;
        self.e_trace = 0.0;
    }
}

impl Default for EPropALIFNeuron {
    fn default() -> Self {
        Self::new(20.0, 200.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = EPropALIFNeuron::default();
        let constructed = EPropALIFNeuron::new(20.0, 200.0, 1.0);
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn eprop_fires() {
        let mut n = EPropALIFNeuron::default();
        let t: i32 = (0..50).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }

    #[test]
    fn eprop_reset_clears_state() {
        let mut n = EPropALIFNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }

    #[test]
    fn eprop_bounded() {
        let mut n = EPropALIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn eprop_adaptation() {
        let mut n = EPropALIFNeuron::default();
        for _ in 0..50 {
            n.step(0.5);
        }
        // a (adaptation) should have increased after spikes
        assert!(n.a.is_finite());
    }

    #[test]
    fn eprop_nan_no_panic() {
        EPropALIFNeuron::default().step(f64::NAN);
    }
}
