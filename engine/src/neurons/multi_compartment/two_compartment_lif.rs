// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Two-compartment LIF neuron model

//! Two-compartment leaky integrate-and-fire neuron model.

/// Two-compartment LIF — soma + dendrite with history-dependent coupling.
#[derive(Clone, Debug)]
pub struct TwoCompartmentLIFNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta: f64,
    pub tau_s: f64,
    pub tau_d: f64,
    pub kappa: f64,
    pub dt: f64,
}

impl TwoCompartmentLIFNeuron {
    pub fn new() -> Self {
        Self {
            v_s: 0.0,
            v_d: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            theta: 1.0,
            tau_s: 2.0,
            tau_d: 20.0,
            kappa: 0.5,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, i_soma: f64, i_dend: f64) -> i32 {
        let alpha_s = (-self.dt / self.tau_s).exp();
        let alpha_d = (-self.dt / self.tau_d).exp();
        self.v_d = alpha_d * self.v_d + i_dend;
        self.v_s = alpha_s * self.v_s + i_soma + self.kappa * self.v_d;
        if self.v_s >= self.theta {
            self.v_s = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = self.v_rest;
        self.v_d = self.v_rest;
    }
}
impl Default for TwoCompartmentLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tc_lif_fires() {
        let mut n = TwoCompartmentLIFNeuron::new();
        let t: i32 = (0..100).map(|_| n.step(0.5, 0.3)).sum();
        assert!(t > 0);
    }

    #[test]
    fn tc_lif_reset() {
        let mut n = TwoCompartmentLIFNeuron::new();
        for _ in 0..50 {
            n.step(0.5, 0.3);
        }
        n.reset();
    }

    #[test]
    fn tc_lif_bounded() {
        let mut n = TwoCompartmentLIFNeuron::new();
        for _ in 0..1000 {
            n.step(100.0, 100.0);
        }
        assert!(n.v_s.is_finite());
    }

    #[test]
    fn tc_lif_nan_no_panic() {
        TwoCompartmentLIFNeuron::new().step(f64::NAN, 0.0);
    }
}
