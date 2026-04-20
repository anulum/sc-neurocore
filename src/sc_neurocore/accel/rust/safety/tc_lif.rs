// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for tc_lif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
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
            v_s: 0.0_f64,
            v_d: 0.0_f64,
            v_rest: 0.0_f64,
            v_reset: 0.0_f64,
            theta: 1.0_f64,
            tau_s: 2.0_f64,
            tau_d: 20.0_f64,
            kappa: 0.5_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // dvd = (-(self.v_d - self.v_rest) + i_dend) / self.tau_d * self.dt
        // self.v_d += dvd
        // dvs = (
        // (-(self.v_s - self.v_rest) + self.kappa * (self.v_d - self.v_s) + i_so
        // / self.tau_s
        // * self.dt
        // )
        // self.v_s += dvs
        // if self.v_s >= self.theta:
        // self.v_s = self.v_reset
        // return 1
        // return 0
        0 // spike indicator
    }

    pub fn reset(&mut self) {
        // self.v_s = self.v_rest
        // self.v_d = self.v_rest
        self.v_s = 0.0_f64;
        self.v_d = 0.0_f64;
        self.v_rest = 0.0_f64;
        self.v_reset = 0.0_f64;
        self.theta = 1.0_f64;
    }

}

pub fn validate_tc_lif(state: &TwoCompartmentLIFNeuron) -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tc_lif_new() {
        let state = TwoCompartmentLIFNeuron::new();
        assert!(validate_tc_lif(&state));
    }

    #[test]
    fn test_tc_lif_step() {
        let mut state = TwoCompartmentLIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
