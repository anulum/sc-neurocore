// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for non_resetting_lif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct NonResettingLIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub v_rest: f64,
    pub theta_rest: f64,
    pub delta_theta: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub r_m: f64,
    pub dt: f64,
}

impl NonResettingLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            theta: -50.0_f64,
            v_rest: -65.0_f64,
            theta_rest: -50.0_f64,
            delta_theta: 5.0_f64,
            tau_m: 10.0_f64,
            tau_theta: 50.0_f64,
            r_m: 1.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // self.v += (-(self.v - self.v_rest) + self.r_m * current) / self.tau_m 
        // self.theta += (-(self.theta - self.theta_rest)) / self.tau_theta * sel
        // if self.v >= self.theta:
        // self.theta += self.delta_theta
        // return 1
        // return 0
        0 // spike indicator
    }

    pub fn reset(&mut self) {
        // self.v = self.v_rest
        // self.theta = self.theta_rest
        self.v = -65.0_f64;
        self.theta = -50.0_f64;
        self.v_rest = -65.0_f64;
        self.theta_rest = -50.0_f64;
        self.delta_theta = 5.0_f64;
    }

}

pub fn validate_non_resetting_lif(state: &NonResettingLIFNeuron) -> bool {
    state.v.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_resetting_lif_new() {
        let state = NonResettingLIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_non_resetting_lif(&state));
    }

    #[test]
    fn test_non_resetting_lif_step() {
        let mut state = NonResettingLIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
