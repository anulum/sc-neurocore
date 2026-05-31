// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for bertram_phantom

const VOLTAGE_MIN: f64 = -250.0;
const VOLTAGE_MAX: f64 = 250.0;
const GATE_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone)]
pub struct BertramPhantomBurster {
    pub v: f64,
    pub s1: f64,
    pub s2: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_s1: f64,
    pub g_s2: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub v_m: f64,
    pub s_m: f64,
    pub v_n: f64,
    pub s_n: f64,
    pub v_s1: f64,
    pub s_s1: f64,
    pub v_s2: f64,
    pub s_s2: f64,
    pub tau_s1: f64,
    pub tau_s2: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl BertramPhantomBurster {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            s1: 0.1,
            s2: 0.1,
            g_ca: 3.6,
            g_k: 10.0,
            g_s1: 4.0,
            g_s2: 4.0,
            g_l: 0.2,
            e_ca: 25.0,
            e_k: -75.0,
            e_l: -40.0,
            c_m: 5.3,
            v_m: -20.0,
            s_m: 12.0,
            v_n: -16.0,
            s_n: 5.6,
            v_s1: -40.0,
            s_s1: 10.0,
            v_s2: -42.0,
            s_s2: 0.4,
            tau_s1: 20000.0,
            tau_s2: 100000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }

    fn valid_state(&self) -> bool {
        self.v.is_finite()
            && (VOLTAGE_MIN..=VOLTAGE_MAX).contains(&self.v)
            && self.s1.is_finite()
            && (0.0..=1.0).contains(&self.s1)
            && self.s2.is_finite()
            && (0.0..=1.0).contains(&self.s2)
            && self.g_ca.is_finite()
            && self.g_ca >= 0.0
            && self.g_k.is_finite()
            && self.g_k >= 0.0
            && self.g_s1.is_finite()
            && self.g_s1 >= 0.0
            && self.g_s2.is_finite()
            && self.g_s2 >= 0.0
            && self.g_l.is_finite()
            && self.g_l >= 0.0
            && self.e_ca.is_finite()
            && self.e_k.is_finite()
            && self.e_l.is_finite()
            && self.c_m.is_finite()
            && self.c_m > 0.0
            && self.v_m.is_finite()
            && self.s_m.is_finite()
            && self.s_m > 0.0
            && self.v_n.is_finite()
            && self.s_n.is_finite()
            && self.s_n > 0.0
            && self.v_s1.is_finite()
            && self.s_s1.is_finite()
            && self.s_s1 > 0.0
            && self.v_s2.is_finite()
            && self.s_s2.is_finite()
            && self.s_s2 > 0.0
            && self.tau_s1.is_finite()
            && self.tau_s1 > 0.0
            && self.tau_s2.is_finite()
            && self.tau_s2 > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }

    pub fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        let z = (vh - v) / k;
        if z >= 0.0 {
            let exp_neg = (-z).exp();
            return exp_neg / (1.0 + exp_neg);
        }
        let exp_pos = z.exp();
        1.0 / (1.0 + exp_pos)
    }

    fn derivatives(&self, v: f64, s1: f64, s2: f64, i_ext: f64) -> (f64, f64, f64) {
        let m_inf = Self::boltz(v, self.v_m, self.s_m);
        let n_inf = Self::boltz(v, self.v_n, self.s_n);
        let s1_inf = Self::boltz(v, self.v_s1, self.s_s1);
        let s2_inf = Self::boltz(v, self.v_s2, self.s_s2);
        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let i_k = self.g_k * n_inf * (v - self.e_k);
        let i_s1 = self.g_s1 * s1 * (v - self.e_k);
        let i_s2 = self.g_s2 * s2 * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_ca - i_k - i_s1 - i_s2 - i_l + i_ext) / self.c_m;
        let ds1 = (s1_inf - s1) / self.tau_s1;
        let ds2 = (s2_inf - s2) / self.tau_s2;
        (dv, ds1, ds2)
    }

    fn candidate_valid(v: f64, s1: f64, s2: f64) -> bool {
        v.is_finite()
            && (VOLTAGE_MIN..=VOLTAGE_MAX).contains(&v)
            && s1.is_finite()
            && (-GATE_TOL..=1.0 + GATE_TOL).contains(&s1)
            && s2.is_finite()
            && (-GATE_TOL..=1.0 + GATE_TOL).contains(&s2)
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.valid_state() {
            return 0;
        }
        let v_prev = self.v;
        let h = self.dt;
        let k1 = self.derivatives(self.v, self.s1, self.s2, i_ext);
        let k2 = self.derivatives(
            self.v + 0.5 * h * k1.0,
            self.s1 + 0.5 * h * k1.1,
            self.s2 + 0.5 * h * k1.2,
            i_ext,
        );
        let k3 = self.derivatives(
            self.v + 0.5 * h * k2.0,
            self.s1 + 0.5 * h * k2.1,
            self.s2 + 0.5 * h * k2.2,
            i_ext,
        );
        let k4 = self.derivatives(
            self.v + h * k3.0,
            self.s1 + h * k3.1,
            self.s2 + h * k3.2,
            i_ext,
        );
        let v = self.v + h * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let s1 = self.s1 + h * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        let s2 = self.s2 + h * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0;
        if !Self::candidate_valid(v, s1, s2) {
            return 0;
        }
        self.v = v;
        self.s1 = s1.clamp(0.0, 1.0);
        self.s2 = s2.clamp(0.0, 1.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -50.0;
        self.s1 = 0.1;
        self.s2 = 0.1;
    }
}

pub fn validate_bertram_phantom(state: &BertramPhantomBurster) -> bool {
    state.valid_state()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bertram_phantom_new() {
        let state = BertramPhantomBurster::new();
        assert!(state.v.is_finite());
        assert!(validate_bertram_phantom(&state));
    }

    #[test]
    fn test_bertram_phantom_rk4_step_updates_all_state() {
        let mut state = BertramPhantomBurster::new();
        let before = (state.v, state.s1, state.s2);
        let spike = state.step(200.0);
        assert!(spike == 0 || spike == 1);
        assert_ne!((state.v, state.s1, state.s2), before);
        assert!(validate_bertram_phantom(&state));
    }

    #[test]
    fn test_invalid_candidate_preserves_state() {
        let mut state = BertramPhantomBurster::new();
        let before = (state.v, state.s1, state.s2);
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!((state.v, state.s1, state.s2), before);
        state.v = f64::NAN;
        assert_eq!(state.step(200.0), 0);
    }
}
