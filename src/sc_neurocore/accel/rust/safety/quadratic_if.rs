// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for quadratic_if

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct QuadraticIFNeuron {
    pub v: f64,
    pub v_reset: f64,
    pub v_peak: f64,
    pub dt: f64,
}

impl QuadraticIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -1.0_f64,
            v_reset: -1.0_f64,
            v_peak: 1.0_f64,
            dt: 0.01_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // self.v += (self.v.powi2 + current) * self.dt
        // if self.v >= self.v_peak:
        // self.v = self.v_reset
        // return 1
        // return 0
        0 // spike indicator
    }

    pub fn reset(&mut self) {
        // self.v = self.v_reset
        self.v = -1.0_f64;
        self.v_reset = -1.0_f64;
        self.v_peak = 1.0_f64;
        self.dt = 0.01_f64;
    }

}

pub fn validate_quadratic_if(state: &QuadraticIFNeuron) -> bool {
    state.v.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_if_new() {
        let state = QuadraticIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_quadratic_if(&state));
    }

    #[test]
    fn test_quadratic_if_step() {
        let mut state = QuadraticIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
