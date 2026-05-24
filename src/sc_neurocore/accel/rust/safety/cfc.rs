// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for cfc

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct ClosedFormContinuousNeuron {
    pub x: f64,
    pub w_tau: f64,
    pub w_x: f64,
    pub w_in: f64,
    pub tau_base: f64,
    pub bias: f64,
    pub v_threshold: f64,
    pub dt: f64,
}

impl ClosedFormContinuousNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0_f64,
            w_tau: -0.5_f64,
            w_x: 0.8_f64,
            w_in: 1.0_f64,
            tau_base: 10.0_f64,
            bias: 0.0_f64,
            v_threshold: 1.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_cfc(self) || !i_ext.is_finite() {
            return 0;
        }

        let gate = self.w_tau * i_ext + self.bias;
        let sigma_tau = if gate >= 0.0 {
            let z = (-gate).exp();
            1.0 / (1.0 + z)
        } else {
            let z = gate.exp();
            z / (1.0 + z)
        };
        let tau_eff = (self.tau_base * sigma_tau).max(0.1);
        let f_target = (self.w_x * self.x + self.w_in * i_ext).tanh();
        let decay = (-self.dt / tau_eff).exp();
        self.x = self.x * decay + f_target * (1.0 - decay);
        if self.x >= self.v_threshold {
            self.x = 0.0;
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        // self.x = 0.0
        self.x = 0.0_f64;
        self.w_tau = -0.5_f64;
        self.w_x = 0.8_f64;
        self.w_in = 1.0_f64;
        self.tau_base = 10.0_f64;
    }
}

pub fn validate_cfc(state: &ClosedFormContinuousNeuron) -> bool {
    state.x.is_finite()
        && state.w_tau.is_finite()
        && state.w_x.is_finite()
        && state.w_in.is_finite()
        && state.tau_base.is_finite()
        && state.tau_base > 0.0
        && state.bias.is_finite()
        && state.v_threshold.is_finite()
        && state.v_threshold > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cfc_new() {
        let state = ClosedFormContinuousNeuron::new();
        assert!(validate_cfc(&state));
    }

    #[test]
    fn test_cfc_step() {
        let mut state = ClosedFormContinuousNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
