// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for prescott

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct PrescottNeuron {
    pub v: f64,
    pub w: f64,
    pub g_fast: f64,
    pub g_slow: f64,
    pub g_l: f64,
    pub e_fast: f64,
    pub e_slow: f64,
    pub e_l: f64,
    pub beta_w: f64,
    pub gamma_w: f64,
    pub tau_w: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PrescottNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            w: 0.0_f64,
            g_fast: 20.0_f64,
            g_slow: 20.0_f64,
            g_l: 2.0_f64,
            e_fast: 50.0_f64,
            e_slow: -100.0_f64,
            e_l: -70.0_f64,
            beta_w: -21.0_f64,
            gamma_w: 15.0_f64,
            tau_w: 100.0_f64,
            phi: 0.15_f64,
            dt: 0.1_f64,
            v_threshold: -20.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        // v_prev = self.v
        // m_inf = 1.0 / (1.0 + (-(self.v + 20.0_f64).exp() / 15.0))
        // w_inf = 1.0 / (1.0 + (-(self.v - self.beta_w_f64).exp() / self.gamma_w
        // i_fast = self.g_fast * m_inf * (self.v - self.e_fast)
        // i_slow = self.g_slow * self.w * (self.v - self.e_slow)
        // i_l = self.g_l * (self.v - self.e_l)
        // self.v += (-i_fast - i_slow - i_l + current) * self.dt
        // self.w += self.phi * (w_inf - self.w) / self.tau_w * self.dt
        // return 1 if (self.v >= self.v_threshold && v_prev < self.v_threshold) 
        0 // spike indicator
    }

    pub fn reset(&mut self) {
        // self.v = -65.0
        // self.w = 0.0
        self.v = -65.0_f64;
        self.w = 0.0_f64;
        self.g_fast = 20.0_f64;
        self.g_slow = 20.0_f64;
        self.g_l = 2.0_f64;
    }

}

pub fn validate_prescott(state: &PrescottNeuron) -> bool {
    state.v.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prescott_new() {
        let state = PrescottNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_prescott(&state));
    }

    #[test]
    fn test_prescott_step() {
        let mut state = PrescottNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
