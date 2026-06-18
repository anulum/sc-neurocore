// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for gutkin_ermentrout

#![allow(dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct GutkinErmentroutNeuron {
    pub v: f64,
    pub n: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl GutkinErmentroutNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            n: 0.1_f64,
            g_na: 20.0_f64,
            g_k: 10.0_f64,
            g_l: 8.0_f64,
            e_na: 60.0_f64,
            e_k: -90.0_f64,
            e_l: -80.0_f64,
            dt: 0.05_f64,
            v_threshold: -20.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_gutkin_ermentrout(self) || !i_ext.is_finite() {
            return -1;
        }
        let v_prev = self.v;
        let Some((k1_v, k1_n)) = self.rhs(self.v, self.n, i_ext) else {
            return -1;
        };
        let Some((k2_v, k2_n)) = self.rhs(
            self.v + 0.5 * self.dt * k1_v,
            self.n + 0.5 * self.dt * k1_n,
            i_ext,
        ) else {
            return -1;
        };
        let Some((k3_v, k3_n)) = self.rhs(
            self.v + 0.5 * self.dt * k2_v,
            self.n + 0.5 * self.dt * k2_n,
            i_ext,
        ) else {
            return -1;
        };
        let Some((k4_v, k4_n)) = self.rhs(self.v + self.dt * k3_v, self.n + self.dt * k3_n, i_ext)
        else {
            return -1;
        };
        let next_v = self.v + self.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        let next_n = self.n + self.dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0;
        if !(next_v.is_finite() && next_n.is_finite() && (0.0..=1.0).contains(&next_n)) {
            return -1;
        }
        self.v = next_v;
        self.n = next_n;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // self.v = -65.0
        // self.n = 0.1
        self.v = -65.0_f64;
        self.n = 0.1_f64;
        self.g_na = 20.0_f64;
        self.g_k = 10.0_f64;
        self.g_l = 8.0_f64;
    }

    fn rhs(&self, v: f64, n_gate: f64, i_ext: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && n_gate.is_finite() && i_ext.is_finite()) {
            return None;
        }
        if !(0.0..=1.0).contains(&n_gate) {
            return None;
        }
        let m_inf = 1.0 / (1.0 + (-(v + 20.0_f64) / 15.0_f64).exp());
        let n_inf = 1.0 / (1.0 + (-(v + 25.0_f64) / 5.0_f64).exp());
        if !(m_inf.is_finite() && n_inf.is_finite()) {
            return None;
        }
        let i_na = self.g_na * m_inf * (v - self.e_na);
        let i_k = self.g_k * n_gate * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = -i_na - i_k - i_l + i_ext;
        let dn = n_inf - n_gate;
        if dv.is_finite() && dn.is_finite() {
            Some((dv, dn))
        } else {
            None
        }
    }
}

pub fn validate_gutkin_ermentrout(state: &GutkinErmentroutNeuron) -> bool {
    state.v.is_finite()
        && state.n.is_finite()
        && (0.0..=1.0).contains(&state.n)
        && state.g_na.is_finite()
        && state.g_na >= 0.0
        && state.g_k.is_finite()
        && state.g_k >= 0.0
        && state.g_l.is_finite()
        && state.g_l >= 0.0
        && state.e_na.is_finite()
        && state.e_k.is_finite()
        && state.e_l.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gutkin_ermentrout_new() {
        let state = GutkinErmentroutNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_gutkin_ermentrout(&state));
    }

    #[test]
    fn test_gutkin_ermentrout_step() {
        let mut state = GutkinErmentroutNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_gutkin_ermentrout_rejects_invalid_without_mutation() {
        let mut state = GutkinErmentroutNeuron::new();
        let v0 = state.v;
        let n0 = state.n;
        assert_eq!(state.step(f64::NAN), -1);
        assert_eq!(state.v, v0);
        assert_eq!(state.n, n0);
    }
}
