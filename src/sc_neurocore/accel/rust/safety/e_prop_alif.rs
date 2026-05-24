// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for e_prop_alif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct EPropALIFNeuron {
    pub v: f64,
    pub a: f64,
    pub e_trace: f64,
    pub tau_m: f64,
    pub tau_a: f64,
    pub v_threshold_base: f64,
    pub beta: f64,
    pub v_reset: f64,
    pub dt: f64,
    pub alpha_m: f64,
    pub alpha_a: f64,
}

impl EPropALIFNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            a: 0.0_f64,
            e_trace: 0.0_f64,
            tau_m: 20.0_f64,
            tau_a: 200.0_f64,
            v_threshold_base: 1.0_f64,
            beta: 0.07_f64,
            v_reset: 0.0_f64,
            dt: 1.0_f64,
            alpha_m: (-1.0_f64 / 20.0_f64).exp(),
            alpha_a: (-1.0_f64 / 200.0_f64).exp(),
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_e_prop_alif(self) || !i_ext.is_finite() {
            return 0;
        }

        self.v = self.alpha_m * self.v + i_ext;
        let threshold = self.v_threshold_base + self.beta * self.a;
        let psi = (1.0 - (self.v - threshold).abs()).max(0.0) * 0.3;
        self.e_trace = self.alpha_a * self.e_trace + psi;
        if self.v >= threshold {
            self.v = self.v_reset;
            self.a = self.alpha_a * self.a + 1.0;
            return 1;
        }
        self.a *= self.alpha_a;
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_reset;
        self.a = 0.0_f64;
        self.e_trace = 0.0_f64;
    }
}

pub fn validate_e_prop_alif(state: &EPropALIFNeuron) -> bool {
    state.v.is_finite()
        && state.a.is_finite()
        && state.a >= 0.0
        && state.e_trace.is_finite()
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.tau_a.is_finite()
        && state.tau_a > 0.0
        && state.v_threshold_base.is_finite()
        && state.beta.is_finite()
        && state.beta >= 0.0
        && state.v_reset.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.dt <= state.tau_m
        && state.dt <= state.tau_a
        && state.v_threshold_base > state.v_reset
        && state.alpha_m.is_finite()
        && state.alpha_m > 0.0
        && state.alpha_m < 1.0
        && state.alpha_a.is_finite()
        && state.alpha_a > 0.0
        && state.alpha_a < 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e_prop_alif_new() {
        let state = EPropALIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_e_prop_alif(&state));
    }

    #[test]
    fn test_e_prop_alif_step() {
        let mut state = EPropALIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_e_prop_alif_reset_uses_v_reset() {
        let mut state = EPropALIFNeuron::new();
        state.v_reset = -0.25;
        state.v = 2.0;
        state.a = 3.0;
        state.e_trace = 4.0;
        state.reset();
        assert_eq!(state.v, -0.25);
        assert_eq!(state.a, 0.0);
        assert_eq!(state.e_trace, 0.0);
    }
}
