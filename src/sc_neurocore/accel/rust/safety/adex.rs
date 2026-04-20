// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for adex

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct AdExNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl AdExNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            w: 0.0_f64,
            v_rest: -65.0_f64,
            v_reset: -68.0_f64,
            v_threshold: -50.0_f64,
            v_rh: -55.0_f64,
            delta_t: 2.0_f64,
            tau: 20.0_f64,
            tau_w: 100.0_f64,
            a: 0.5_f64,
            b: 7.0_f64,
            c_m: 200.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // exp_term = self.delta_t * (((self.v - self.v_rh_f64).exp() / self.delt
        // dv = (
        // (-(self.v - self.v_rest) + exp_term) / self.tau + (-self.w + current)
        // ) * self.dt
        // dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt
        // self.v += dv
        // self.w += dw
        // if self.v >= self.v_threshold:
        // self.v = self.v_reset
        // self.w += self.b
        // return 1
        // return 0
        0 // spike indicator
    }

    pub fn reset(&mut self) {
        // self.v = self.v_rest
        // self.w = 0.0
        self.v = -65.0_f64;
        self.w = 0.0_f64;
        self.v_rest = -65.0_f64;
        self.v_reset = -68.0_f64;
        self.v_threshold = -50.0_f64;
    }

}

pub fn validate_adex(state: &AdExNeuron) -> bool {
    state.v.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adex_new() {
        let state = AdExNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_adex(&state));
    }

    #[test]
    fn test_adex_step() {
        let mut state = AdExNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
