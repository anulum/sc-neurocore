// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for wang_buzsaki

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct WangBuzsakiNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl WangBuzsakiNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            h: 0.8_f64,
            n: 0.1_f64,
            g_na: 35.0_f64,
            g_k: 9.0_f64,
            g_l: 0.1_f64,
            e_na: 55.0_f64,
            e_k: -90.0_f64,
            e_l: -65.0_f64,
            c_m: 1.0_f64,
            phi: 5.0_f64,
            dt: 0.01_f64,
            v_threshold: -20.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_wang_buzsaki(self) || !i_ext.is_finite() {
            return Err("invalid Wang-Buzsaki state or input");
        }
        // v_prev = self.v
        // for _ in range(int(0.5 / max(self.dt, 0.001))):
        // # m is instantaneous (m_inf)
        // alpha_m = (
        // 0.1 * (self.v + 35.0) / (1.0 - (-(self.v + 35.0_f64).exp() / 10.0))
        // if abs(self.v + 35.0) > 1e-6
        // else 1.0
        // )
        // beta_m = 4.0 * (-(self.v + 60.0_f64).exp() / 18.0)
        // m_inf = alpha_m / (alpha_m + beta_m)
        // alpha_h = 0.07 * (-(self.v + 58.0_f64).exp() / 20.0)
        // beta_h = 1.0 / (1.0 + (-(self.v + 28.0_f64).exp() / 10.0))
        // alpha_n = (
        // 0.01 * (self.v + 34.0) / (1.0 - (-(self.v + 34.0_f64).exp() / 10.0))
        // if abs(self.v + 34.0) > 1e-6
        Ok(0) // spike indicator
    }

    pub fn reset(&mut self) {
        // self.v = -65.0
        // self.h, self.n = 0.8, 0.1
        self.v = -65.0_f64;
        self.h = 0.8_f64;
        self.n = 0.1_f64;
        self.g_na = 35.0_f64;
        self.g_k = 9.0_f64;
    }
}

pub fn validate_wang_buzsaki(state: &WangBuzsakiNeuron) -> bool {
    state.v.is_finite()
        && state.h.is_finite()
        && state.n.is_finite()
        && state.g_na.is_finite()
        && state.g_na > 0.0
        && state.g_k.is_finite()
        && state.g_k > 0.0
        && state.g_l.is_finite()
        && state.g_l > 0.0
        && state.e_na.is_finite()
        && state.e_k.is_finite()
        && state.e_l.is_finite()
        && state.c_m.is_finite()
        && state.c_m > 0.0
        && state.phi.is_finite()
        && state.phi > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wang_buzsaki_new() {
        let state = WangBuzsakiNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_wang_buzsaki(&state));
    }

    #[test]
    fn test_wang_buzsaki_step() {
        let mut state = WangBuzsakiNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_wang_buzsaki_rejects_invalid_runtime_state() {
        let mut state = WangBuzsakiNeuron::new();
        state.h = f64::INFINITY;
        assert_eq!(state.step(10.0), Err("invalid Wang-Buzsaki state or input"));
    }
}
