// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for wilson_cowan

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct WilsonCowanUnit {
    pub e: f64,
    pub i: f64,
    pub w_ee: f64,
    pub w_ei: f64,
    pub w_ie: f64,
    pub w_ii: f64,
    pub tau_e: f64,
    pub tau_i: f64,
    pub a: f64,
    pub theta: f64,
    pub dt: f64,
}

impl WilsonCowanUnit {
    pub fn new() -> Self {
        Self {
            e: 0.1_f64,
            i: 0.05_f64,
            w_ee: 10.0_f64,
            w_ei: 6.0_f64,
            w_ie: 10.0_f64,
            w_ii: 1.0_f64,
            tau_e: 1.0_f64,
            tau_i: 2.0_f64,
            a: 1.2_f64,
            theta: 4.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn _sigmoid(&self, x: f64) -> f64 {
        logistic(self.a * (x - self.theta)) - logistic(-self.a * self.theta)
    }

    pub fn step(&mut self, i_ext: f64) -> Result<f64, &'static str> {
        if !validate_wilson_cowan(self) {
            return Err("invalid Wilson-Cowan runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Wilson-Cowan external input");
        }

        let se = self._sigmoid(self.w_ee * self.e - self.w_ei * self.i + i_ext);
        let si = self._sigmoid(self.w_ie * self.e - self.w_ii * self.i);
        if !se.is_finite() || !si.is_finite() {
            return Err("invalid Wilson-Cowan sigmoid output");
        }
        let next_e = self.e + (-self.e + se) / self.tau_e * self.dt;
        let next_i = self.i + (-self.i + si) / self.tau_i * self.dt;
        if !finite_rate(next_e, self.a, self.theta) || !finite_rate(next_i, self.a, self.theta) {
            return Err("invalid Wilson-Cowan candidate state");
        }

        self.e = next_e;
        self.i = next_i;
        Ok(self.e)
    }

    pub fn reset(&mut self) {
        // self.e, self.i = 0.1, 0.05
        self.e = 0.1_f64;
        self.i = 0.05_f64;
        self.w_ee = 10.0_f64;
        self.w_ei = 6.0_f64;
        self.w_ie = 10.0_f64;
    }
}

pub fn validate_wilson_cowan(state: &WilsonCowanUnit) -> bool {
    state.w_ee.is_finite()
        && state.w_ee >= 0.0
        && state.w_ei.is_finite()
        && state.w_ei >= 0.0
        && state.w_ie.is_finite()
        && state.w_ie >= 0.0
        && state.w_ii.is_finite()
        && state.w_ii >= 0.0
        && state.tau_e.is_finite()
        && state.tau_e > 0.0
        && state.tau_i.is_finite()
        && state.tau_i > 0.0
        && state.a.is_finite()
        && state.a > 0.0
        && state.theta.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && finite_rate(state.e, state.a, state.theta)
        && finite_rate(state.i, state.a, state.theta)
}

fn finite_rate(value: f64, a: f64, theta: f64) -> bool {
    let baseline = logistic(-a * theta);
    value.is_finite() && value >= -baseline && value <= 1.0 - baseline
}

fn logistic(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let exp_z = z.exp();
        exp_z / (1.0 + exp_z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wilson_cowan_new() {
        let state = WilsonCowanUnit::new();
        assert!(validate_wilson_cowan(&state));
    }

    #[test]
    fn test_wilson_cowan_step() {
        let mut state = WilsonCowanUnit::new();
        let rate = state.step(10.0).unwrap();
        assert!(rate.is_finite());
    }

    #[test]
    fn test_wilson_cowan_rejects_invalid_runtime_state() {
        let mut state = WilsonCowanUnit::new();
        state.e = 1.5;
        assert!(state.step(1.0).is_err());
    }
}
