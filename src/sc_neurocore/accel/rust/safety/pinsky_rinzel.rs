// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for pinsky_rinzel

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct PinskyRinzelNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub h: f64,
    pub n: f64,
    pub s: f64,
    pub c: f64,
    pub q: f64,
    pub gc: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_kdr: f64,
    pub g_ca: f64,
    pub g_kahp: f64,
    pub g_kc: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PinskyRinzelNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -60.0_f64,
            v_d: -60.0_f64,
            h: 0.9_f64,
            n: 0.1_f64,
            s: 0.0_f64,
            c: 0.0_f64,
            q: 0.0_f64,
            gc: 2.1_f64,
            p: 0.5_f64,
            g_na: 30.0_f64,
            g_kdr: 15.0_f64,
            g_ca: 10.0_f64,
            g_kahp: 0.8_f64,
            g_kc: 15.0_f64,
            g_l: 0.1_f64,
            e_na: 60.0_f64,
            e_k: -75.0_f64,
            e_ca: 80.0_f64,
            e_l: -60.0_f64,
            dt: 0.02_f64,
            v_threshold: -20.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        self.step_dend(i_ext, 0.0)
    }

    pub fn step_dend(&mut self, current_soma: f64, current_dend: f64) -> i32 {
        if !(validate_pinsky_rinzel(self) && current_soma.is_finite() && current_dend.is_finite()) {
            return -1;
        }
        let v_prev = self.v_s;
        let am = alpha(0.32, self.v_s + 54.0, 4.0, 8.0, false);
        let bm = alpha(0.28, self.v_s + 27.0, 5.0, 5.6, true);
        let m_inf = am / (am + bm);
        let ah = 0.128 * (-(self.v_s + 50.0) / 18.0).exp();
        let bh = 4.0 * logistic((self.v_s + 27.0) / 5.0);
        let an = alpha(0.032, self.v_s + 52.0, 5.0, 0.32, false);
        let bn = 0.5 * (-(self.v_s + 57.0) / 40.0).exp();
        let s_inf = logistic((self.v_d + 20.0) / 9.0);
        let i_na = self.g_na * m_inf.powi(2) * self.h * (self.v_s - self.e_na);
        let i_kdr = self.g_kdr * self.n * (self.v_s - self.e_k);
        let i_ls = self.g_l * (self.v_s - self.e_l);
        let i_ds = (self.gc / self.p) * (self.v_s - self.v_d);
        let i_ca = self.g_ca * self.s.powi(2) * (self.v_d - self.e_ca);
        let i_kahp = self.g_kahp * self.q * (self.v_d - self.e_k);
        let chi = if self.v_d <= 50.0 {
            (self.v_d / 250.0 + 0.5).min(1.0)
        } else {
            2.0
        };
        let i_kc = self.g_kc * self.c * chi * (self.v_d - self.e_k);
        let i_ld = self.g_l * (self.v_d - self.e_l);
        let i_sd = (self.gc / (1.0 - self.p)) * (self.v_d - self.v_s);

        let mut next = self.clone();
        next.v_s += (-i_na - i_kdr - i_ls - i_ds + current_soma / self.p) * self.dt;
        next.v_d += (-i_ca - i_kahp - i_kc - i_ld - i_sd + current_dend / (1.0 - self.p)) * self.dt;
        next.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
        next.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
        next.s += ((s_inf - self.s) / 5.0) * self.dt;
        next.c = (self.c + (-0.13 * i_ca - 0.075 * self.c) * self.dt).max(0.0);
        let q_inf = (next.c / (next.c + 2.0)).min(1.0);
        next.q += ((q_inf - self.q) / 100.0) * self.dt;
        if !validate_pinsky_rinzel(&next) {
            return -1;
        }
        *self = next;
        if self.v_s >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        // self.v_s, self.v_d = -60.0, -60.0
        // self.h, self.n, self.s, self.c, self.q = 0.9, 0.1, 0.0, 0.0, 0.0
        self.v_s = -60.0_f64;
        self.v_d = -60.0_f64;
        self.h = 0.9_f64;
        self.n = 0.1_f64;
        self.s = 0.0_f64;
        self.c = 0.0_f64;
        self.q = 0.0_f64;
        self.gc = 2.1_f64;
        self.p = 0.5_f64;
        self.g_na = 30.0_f64;
        self.g_kdr = 15.0_f64;
        self.g_ca = 10.0_f64;
        self.g_kahp = 0.8_f64;
        self.g_kc = 15.0_f64;
        self.g_l = 0.1_f64;
        self.e_na = 60.0_f64;
        self.e_k = -75.0_f64;
        self.e_ca = 80.0_f64;
        self.e_l = -60.0_f64;
        self.dt = 0.02_f64;
        self.v_threshold = -20.0_f64;
    }
}

pub fn validate_pinsky_rinzel(state: &PinskyRinzelNeuron) -> bool {
    state.v_s.is_finite()
        && state.v_d.is_finite()
        && gate(state.h)
        && gate(state.n)
        && gate(state.s)
        && state.c.is_finite()
        && state.c >= 0.0
        && gate(state.q)
        && state.gc.is_finite()
        && state.gc > 0.0
        && state.p.is_finite()
        && state.p > 0.0
        && state.p < 1.0
        && state.g_na.is_finite()
        && state.g_na > 0.0
        && state.g_kdr.is_finite()
        && state.g_kdr > 0.0
        && state.g_ca.is_finite()
        && state.g_ca > 0.0
        && state.g_kahp.is_finite()
        && state.g_kahp > 0.0
        && state.g_kc.is_finite()
        && state.g_kc > 0.0
        && state.g_l.is_finite()
        && state.g_l > 0.0
        && state.e_na.is_finite()
        && state.e_k.is_finite()
        && state.e_ca.is_finite()
        && state.e_l.is_finite()
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.v_threshold.is_finite()
}

fn gate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn alpha(scale: f64, x: f64, divisor: f64, fallback: f64, positive_exp: bool) -> f64 {
    if x.abs() <= 1e-6 {
        return fallback;
    }
    if positive_exp {
        scale * x / ((x / divisor).exp() - 1.0)
    } else {
        scale * x / (1.0 - (-x / divisor).exp())
    }
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinsky_rinzel_new() {
        let state = PinskyRinzelNeuron::new();
        assert!(validate_pinsky_rinzel(&state));
    }

    #[test]
    fn test_pinsky_rinzel_step() {
        let mut state = PinskyRinzelNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_pinsky_rinzel_rejects_invalid_without_mutation() {
        let mut state = PinskyRinzelNeuron::new();
        state.p = 1.0;
        let before = state.clone();
        assert_eq!(state.step(10.0), -1);
        assert_eq!(state.v_s, before.v_s);
        assert_eq!(state.v_d, before.v_d);
    }
}
