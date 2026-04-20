// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for nlif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct NonlinearLIFNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_crit: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub a: f64,
    pub b: f64,
    pub tau_w: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl NonlinearLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            w: 0.0_f64,
            v_rest: -65.0_f64,
            v_crit: -40.0_f64,
            v_threshold: -20.0_f64,
            v_reset: -65.0_f64,
            a: 0.04_f64,
            b: 0.5_f64,
            tau_w: 100.0_f64,
            c_m: 1.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // cubic = self.a * (self.v - self.v_rest) * (self.v - self.v_crit)
        // dv = (cubic - self.w + current) / self.c_m * self.dt
        // dw = (self.b * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt
        // self.v += dv
        // self.w += dw
        // if self.v >= self.v_threshold:
        // self.v = self.v_reset
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
        self.v_crit = -40.0_f64;
        self.v_threshold = -20.0_f64;
    }

}

pub fn validate_nlif(state: &NonlinearLIFNeuron) -> bool {
    state.v.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nlif_new() {
        let state = NonlinearLIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_nlif(&state));
    }

    #[test]
    fn test_nlif_step() {
        let mut state = NonlinearLIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
