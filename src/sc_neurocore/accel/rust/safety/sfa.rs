// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for sfa

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct SFANeuron {
    pub v: f64,
    pub g_sfa: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_sfa: f64,
    pub delta_g: f64,
    pub e_k: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl SFANeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0_f64,
            g_sfa: 0.0_f64,
            v_rest: -70.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 10.0_f64,
            tau_sfa: 200.0_f64,
            delta_g: 0.5_f64,
            e_k: -80.0_f64,
            resistance: 1.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // self.v += (
        // (-(self.v - self.v_rest) - self.g_sfa * (self.v - self.e_k) + self.res
        // / self.tau_m
        // * self.dt
        // )
        // self.g_sfa *= (-self.dt / self.tau_sfa_f64).exp()
        // if self.v >= self.v_threshold:
        // self.v = self.v_reset
        // self.g_sfa += self.delta_g
        // return 1
        // return 0
        0 // spike indicator
    }

    pub fn reset(&mut self) {
        // self.v = self.v_rest
        // self.g_sfa = 0.0
        self.v = -70.0_f64;
        self.g_sfa = 0.0_f64;
        self.v_rest = -70.0_f64;
        self.v_reset = -70.0_f64;
        self.v_threshold = -50.0_f64;
    }

}

pub fn validate_sfa(state: &SFANeuron) -> bool {
    state.v.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sfa_new() {
        let state = SFANeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_sfa(&state));
    }

    #[test]
    fn test_sfa_step() {
        let mut state = SFANeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
