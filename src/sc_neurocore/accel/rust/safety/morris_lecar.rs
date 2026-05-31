// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for morris_lecar

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct MorrisLecarNeuron {
    pub v: f64,
    pub w: f64,
    pub c_m: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl MorrisLecarNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0_f64,
            w: 0.0_f64,
            c_m: 20.0_f64,
            g_ca: 4.0_f64,
            g_k: 8.0_f64,
            g_l: 2.0_f64,
            e_ca: 120.0_f64,
            e_k: -84.0_f64,
            e_l: -60.0_f64,
            v1: -1.2_f64,
            v2: 18.0_f64,
            v3: 12.0_f64,
            v4: 17.4_f64,
            phi: 1.0_f64 / 15.0_f64,
            dt: 0.1_f64,
            v_threshold: 0.0_f64,
        }
    }

    pub fn _m_inf(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - self.v1) / self.v2).tanh())
    }

    pub fn _w_inf(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - self.v3) / self.v4).tanh())
    }

    pub fn _lam(&self, v: f64) -> f64 {
        self.phi * ((v - self.v3) / (2.0 * self.v4)).cosh()
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !validate_morris_lecar(self) || !current.is_finite() {
            return -1;
        }
        let v_prev = self.v;
        let m_inf = self._m_inf(self.v);
        let w_inf = self._w_inf(self.v);
        let lam = self._lam(self.v);
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.w * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        let mut next = self.clone();
        next.v += (-i_ca - i_k - i_l + current) / self.c_m * self.dt;
        next.w += lam * (w_inf - self.w) * self.dt;
        if !validate_morris_lecar(&next) {
            return -1;
        }
        *self = next;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // self.v = -60.0
        // self.w = 0.0
        self.v = -60.0_f64;
        self.w = 0.0_f64;
        self.c_m = 20.0_f64;
        self.g_ca = 4.0_f64;
        self.g_k = 8.0_f64;
        self.g_l = 2.0_f64;
        self.e_ca = 120.0_f64;
        self.e_k = -84.0_f64;
        self.e_l = -60.0_f64;
        self.v1 = -1.2_f64;
        self.v2 = 18.0_f64;
        self.v3 = 12.0_f64;
        self.v4 = 17.4_f64;
        self.phi = 1.0_f64 / 15.0_f64;
        self.dt = 0.1_f64;
        self.v_threshold = 0.0_f64;
    }
}

pub fn validate_morris_lecar(state: &MorrisLecarNeuron) -> bool {
    state.v.is_finite()
        && state.w.is_finite()
        && state.c_m.is_finite()
        && state.g_ca.is_finite()
        && state.g_k.is_finite()
        && state.g_l.is_finite()
        && state.e_ca.is_finite()
        && state.e_k.is_finite()
        && state.e_l.is_finite()
        && state.v1.is_finite()
        && state.v2.is_finite()
        && state.v3.is_finite()
        && state.v4.is_finite()
        && state.phi.is_finite()
        && state.dt.is_finite()
        && state.v_threshold.is_finite()
        && state.c_m > 0.0
        && state.g_ca > 0.0
        && state.g_k > 0.0
        && state.g_l > 0.0
        && state.v2 > 0.0
        && state.v4 > 0.0
        && state.phi > 0.0
        && state.dt > 0.0
        && (0.0..=1.0).contains(&state.w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morris_lecar_new() {
        let state = MorrisLecarNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_morris_lecar(&state));
    }

    #[test]
    fn test_morris_lecar_step() {
        let mut state = MorrisLecarNeuron::new();
        let v0 = state.v;
        let w0 = state.w;
        let current = 50.0;
        let m_inf = 0.5 * (1.0 + ((v0 - state.v1) / state.v2).tanh());
        let w_inf = 0.5 * (1.0 + ((v0 - state.v3) / state.v4).tanh());
        let lam = state.phi * ((v0 - state.v3) / (2.0 * state.v4)).cosh();
        let i_ca = state.g_ca * m_inf * (v0 - state.e_ca);
        let i_k = state.g_k * w0 * (v0 - state.e_k);
        let i_l = state.g_l * (v0 - state.e_l);
        let expected_v = v0 + (-i_ca - i_k - i_l + current) / state.c_m * state.dt;
        let expected_w = w0 + lam * (w_inf - w0) * state.dt;

        let spike = state.step(current);

        assert!(spike == 0 || spike == 1);
        assert!((state.v - expected_v).abs() < 1e-12);
        assert!((state.w - expected_w).abs() < 1e-12);
    }

    #[test]
    fn test_morris_lecar_rejects_invalid_state() {
        let mut state = MorrisLecarNeuron::new();
        state.c_m = 0.0;
        let before = state.clone();
        assert_eq!(state.step(50.0), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.w, before.w);
    }
}
