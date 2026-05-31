// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for cochlear_hair_cell

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct CochlearHairCell {
    pub g_max: f64,
    pub e_met: f64,
    pub g_l: f64,
    pub e_l: f64,
    pub cap: f64,
    pub x0: f64,
    pub delta: f64,
    pub dt: f64,
    pub v: f64,
    pub glutamate_release: f64,
}

impl CochlearHairCell {
    pub fn new() -> Self {
        Self {
            g_max: 10.0_f64,
            e_met: 0.0_f64,
            g_l: 1.0_f64,
            e_l: -60.0_f64,
            cap: 10.0_f64,
            x0: 0.0_f64,
            delta: 0.1_f64,
            dt: 0.01_f64,
            v: -60.0_f64,
            glutamate_release: 0.0_f64,
        }
    }

    pub fn p_open(&self, displacement: f64) -> f64 {
        let z = (displacement - self.x0) / self.delta;
        if z >= 0.0 {
            1.0 / (1.0 + (-z).exp())
        } else {
            let ez = z.exp();
            ez / (1.0 + ez)
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_cochlear_hair_cell(self) || !i_ext.is_finite() {
            return -1;
        }
        let po = self.p_open(i_ext);
        let g_met = self.g_max * po;
        let g_total = self.g_l + g_met;
        if !(g_total.is_finite() && g_total > 0.0) {
            return -1;
        }
        let v_inf = (self.g_l * self.e_l + g_met * self.e_met) / g_total;
        let candidate_v = v_inf + (self.v - v_inf) * (-(g_total / self.cap) * self.dt).exp();
        let candidate_release = (candidate_v + 60.0).max(0.0) / 40.0;
        if !(candidate_v.is_finite() && candidate_release.is_finite()) {
            return -1;
        }
        self.v = candidate_v;
        self.glutamate_release = candidate_release;
        if self.glutamate_release > 0.5 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.glutamate_release = 0.0;
    }
}

pub fn validate_cochlear_hair_cell(state: &CochlearHairCell) -> bool {
    [
        state.g_max,
        state.e_met,
        state.g_l,
        state.e_l,
        state.cap,
        state.x0,
        state.delta,
        state.dt,
        state.v,
        state.glutamate_release,
    ]
    .iter()
    .all(|x| x.is_finite())
        && state.g_max >= 0.0
        && state.g_l > 0.0
        && state.cap > 0.0
        && state.delta > 0.0
        && state.dt > 0.0
        && state.glutamate_release >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cochlear_hair_cell_new() {
        let state = CochlearHairCell::new();
        assert!(state.v.is_finite());
        assert!(validate_cochlear_hair_cell(&state));
    }

    #[test]
    fn test_cochlear_hair_cell_step() {
        let mut state = CochlearHairCell::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn closed_form_relaxation() {
        let mut state = CochlearHairCell::new();
        let po = 1.0 / (1.0 + (-(0.0 - state.x0) / state.delta).exp());
        let g_met = state.g_max * po;
        let g_total = state.g_l + g_met;
        let v_inf = (state.g_l * state.e_l + g_met * state.e_met) / g_total;
        let expected = v_inf + (state.v - v_inf) * (-(g_total / state.cap) * state.dt).exp();
        let spike = state.step(0.0);
        assert!(spike == 0 || spike == 1);
        assert!((state.v - expected).abs() < 1e-12);
    }

    #[test]
    fn invalid_runtime_preserves_state() {
        let mut state = CochlearHairCell::new();
        state.v = -55.0;
        state.glutamate_release = 0.125;
        let before = (state.v, state.glutamate_release);
        state.cap = -1.0;
        assert_eq!(state.step(0.25), -1);
        assert_eq!((state.v, state.glutamate_release), before);
    }
}
