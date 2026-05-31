// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for golgi_cell

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct GolgiCell {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub p_na: f64,
    pub n: f64,
    pub a: f64,
    pub b: f64,
    pub w: f64,
    pub m_t: f64,
    pub s: f64,
    pub c_n: f64,
    pub r: f64,
    pub ca: f64,
    pub g_na_t: f64,
    pub g_na_p: f64,
    pub g_kdr: f64,
    pub g_ka: f64,
    pub g_km: f64,
    pub g_cat: f64,
    pub g_can: f64,
    pub g_bk: f64,
    pub g_sk: f64,
    pub g_h: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub tau_ca: f64,
    pub kd_bk: f64,
    pub kd_sk: f64,
    pub dt: f64,
    pub sub_steps: usize,
    pub gain: f64,
}

impl GolgiCell {
    pub fn new() -> Self {
        Self {
            v: -60.0_f64,
            m: 0.02_f64,
            h: 0.85_f64,
            p_na: 0.01_f64,
            n: 0.05_f64,
            a: 0.1_f64,
            b: 0.8_f64,
            w: 0.01_f64,
            m_t: 0.01_f64,
            s: 0.9_f64,
            c_n: 0.01_f64,
            r: 0.1_f64,
            ca: 0.05_f64,
            g_na_t: 48.0_f64,
            g_na_p: 0.2_f64,
            g_kdr: 16.0_f64,
            g_ka: 8.0_f64,
            g_km: 1.0_f64,
            g_cat: 0.5_f64,
            g_can: 1.0_f64,
            g_bk: 3.0_f64,
            g_sk: 1.0_f64,
            g_h: 0.1_f64,
            g_l: 0.05_f64,
            e_na: 55.0_f64,
            e_k: -90.0_f64,
            e_ca: 120.0_f64,
            e_h: -40.0_f64,
            e_l: -55.0_f64,
            c_m: 1.0_f64,
            tau_ca: 200.0_f64,
            kd_bk: 1.0_f64,
            kd_sk: 0.5_f64,
            dt: 0.5_f64,
            sub_steps: 10,
            gain: 1.0_f64,
        }
    }

    fn safe_rate(a: f64, vhalf: f64, v: f64, k: f64, fallback: f64) -> f64 {
        let d = v + vhalf;
        if d.abs() < 1e-7 {
            fallback
        } else {
            a * d / (1.0 - (-d / k).exp())
        }
    }

    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        let x = (v - vh) / k;
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let ex = x.exp();
            ex / (1.0 + ex)
        }
    }

    fn voltage_valid(value: f64) -> bool {
        value.is_finite() && (-100.0..=60.0).contains(&value)
    }

    fn probability(value: f64) -> bool {
        value.is_finite() && (0.0..=1.0).contains(&value)
    }

    fn gate_alpha_beta(previous: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> Option<f64> {
        let total = phi * (alpha + beta);
        if !previous.is_finite()
            || !alpha.is_finite()
            || !beta.is_finite()
            || !total.is_finite()
            || !dt.is_finite()
            || total <= 0.0
        {
            return None;
        }
        let steady = alpha / (alpha + beta);
        Some((steady + (previous - steady) * (-total * dt).exp()).clamp(0.0, 1.0))
    }

    fn gate_inf(previous: f64, steady: f64, tau: f64, dt: f64) -> Option<f64> {
        if !previous.is_finite()
            || !steady.is_finite()
            || !tau.is_finite()
            || !dt.is_finite()
            || tau <= 0.0
        {
            return None;
        }
        Some((steady + (previous - steady) * (-dt / tau).exp()).clamp(0.0, 1.0))
    }

    fn calcium_exact(previous: f64, entry: f64, tau: f64, dt: f64) -> Option<f64> {
        if !previous.is_finite()
            || !entry.is_finite()
            || !tau.is_finite()
            || !dt.is_finite()
            || tau <= 0.0
            || previous < 0.0
        {
            return None;
        }
        let steady = entry * tau;
        let value = steady + (previous - steady) * (-dt / tau).exp();
        value.is_finite().then_some(value.max(0.0))
    }

    fn valid_state(&self) -> bool {
        Self::voltage_valid(self.v)
            && [
                self.m, self.h, self.p_na, self.n, self.a, self.b, self.w, self.m_t, self.s,
                self.c_n, self.r,
            ]
            .iter()
            .all(|value| Self::probability(*value))
            && [
                self.g_na_t,
                self.g_na_p,
                self.g_kdr,
                self.g_ka,
                self.g_km,
                self.g_cat,
                self.g_can,
                self.g_bk,
                self.g_sk,
                self.g_h,
                self.g_l,
            ]
            .iter()
            .all(|g| g.is_finite() && *g >= 0.0)
            && self.ca.is_finite()
            && self.ca >= 0.0
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_ca.is_finite()
            && self.e_h.is_finite()
            && self.e_l.is_finite()
            && self.c_m.is_finite()
            && self.tau_ca.is_finite()
            && self.kd_bk.is_finite()
            && self.kd_sk.is_finite()
            && self.dt.is_finite()
            && self.gain.is_finite()
            && self.c_m > 0.0
            && self.tau_ca > 0.0
            && self.kd_bk > 0.0
            && self.kd_sk > 0.0
            && self.dt > 0.0
            && self.sub_steps > 0
            && self.gain >= 0.0
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !i_ext.is_finite() || !self.valid_state() {
            return 0;
        }

        let input = self.gain * i_ext;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev = self.v;
        let mut next = self.clone();
        for _ in 0..self.sub_steps {
            let v = next.v;
            let alpha_m = Self::safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
            let Some(m) = Self::gate_alpha_beta(next.m, alpha_m, beta_m, 5.0, dt_sub) else {
                return 0;
            };
            let Some(h) = Self::gate_alpha_beta(next.h, alpha_h, beta_h, 5.0, dt_sub) else {
                return 0;
            };
            let tau_pna = 5.0 + 20.0 / (1.0 + ((v + 48.0) / 10.0).powi(2)).max(0.01);
            let Some(p_na) = Self::gate_inf(next.p_na, Self::boltz(v, -48.0, 5.0), tau_pna, dt_sub)
            else {
                return 0;
            };
            let alpha_n = Self::safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();
            let Some(n) = Self::gate_alpha_beta(next.n, alpha_n, beta_n, 5.0, dt_sub) else {
                return 0;
            };
            let Some(a) = Self::gate_inf(next.a, Self::boltz(v, -27.0, 16.0), 2.0, dt_sub) else {
                return 0;
            };
            let Some(b) = Self::gate_inf(next.b, Self::boltz(v, -80.0, -6.0), 15.0, dt_sub) else {
                return 0;
            };
            let tau_w = 100.0 / (3.3 * ((v + 35.0) / 20.0).exp() + (-(v + 35.0) / 20.0).exp());
            let Some(w) = Self::gate_inf(next.w, Self::boltz(v, -35.0, 10.0), tau_w, dt_sub) else {
                return 0;
            };
            let Some(m_t) = Self::gate_inf(next.m_t, Self::boltz(v, -52.0, 5.0), 1.0, dt_sub)
            else {
                return 0;
            };
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).powi(2)).max(0.01);
            let Some(s) = Self::gate_inf(next.s, Self::boltz(v, -60.0, -6.5), tau_s, dt_sub) else {
                return 0;
            };
            let tau_cn = 2.0 + 10.0 / (1.0 + ((v + 20.0) / 10.0).powi(2)).max(0.01);
            let Some(c_n) = Self::gate_inf(next.c_n, Self::boltz(v, -20.0, 5.0), tau_cn, dt_sub)
            else {
                return 0;
            };
            let tau_r = 50.0 + 200.0 / (1.0 + ((v + 80.0) / 20.0).powi(2)).max(0.01);
            let Some(r) = Self::gate_inf(next.r, Self::boltz(v, -80.0, -10.0), tau_r, dt_sub)
            else {
                return 0;
            };

            let g_cat = self.g_cat * m_t.powi(2) * s;
            let g_can = self.g_can * c_n.powi(2);
            let i_ca = g_cat * (v - self.e_ca) + g_can * (v - self.e_ca);
            let ca_entry = if i_ca < 0.0 { -i_ca * 0.001 } else { 0.0 };
            let Some(ca) = Self::calcium_exact(next.ca, ca_entry, self.tau_ca, dt_sub) else {
                return 0;
            };
            let ca2 = ca * ca;
            let bk_v = Self::boltz(v, 100.0 - 120.0 * ca2 / (ca2 + self.kd_bk.powi(2)), 15.0);
            let sk_inf = ca2 / (ca2 + self.kd_sk.powi(2));
            let g_na = self.g_na_t * m.powi(3) * h + self.g_na_p * p_na;
            let g_k = self.g_kdr * n.powi(4)
                + self.g_ka * a.powi(3) * b
                + self.g_km * w
                + self.g_bk * bk_v
                + self.g_sk * sk_inf;
            let g_ca = g_cat + g_can;
            let g_h = self.g_h * r;
            let g_total = g_na + g_k + g_ca + g_h + self.g_l;
            if !g_total.is_finite() || g_total <= 0.0 {
                return 0;
            }
            let steady_v = (input
                + g_na * self.e_na
                + g_k * self.e_k
                + g_ca * self.e_ca
                + g_h * self.e_h
                + self.g_l * self.e_l)
                / g_total;
            let v_next = steady_v + (v - steady_v) * (-(g_total / self.c_m) * dt_sub).exp();
            if !Self::voltage_valid(v_next) || !ca.is_finite() || ca < 0.0 {
                return 0;
            }

            next.v = v_next;
            next.m = m;
            next.h = h;
            next.p_na = p_na;
            next.n = n;
            next.a = a;
            next.b = b;
            next.w = w;
            next.m_t = m_t;
            next.s = s;
            next.c_n = c_n;
            next.r = r;
            next.ca = ca;
        }

        *self = next;
        if self.v >= 0.0 && v_prev < 0.0 {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

pub fn validate_golgi_cell(state: &GolgiCell) -> bool {
    state.valid_state()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golgi_cell_new() {
        let state = GolgiCell::new();
        assert!(state.v.is_finite());
        assert!(validate_golgi_cell(&state));
    }

    #[test]
    fn test_golgi_cell_step() {
        let mut state = GolgiCell::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    fn snapshot(state: &GolgiCell) -> (f64, f64, f64, f64, f64, f64) {
        (state.v, state.m, state.h, state.p_na, state.n, state.ca)
    }

    #[test]
    fn test_golgi_cell_invalid_current_preserves_state() {
        let mut state = GolgiCell::new();
        for _ in 0..10 {
            state.step(5.0);
        }
        let before = snapshot(&state);

        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(snapshot(&state), before);
        assert_eq!(state.step(f64::INFINITY), 0);
        assert_eq!(snapshot(&state), before);
    }

    #[test]
    fn test_golgi_cell_excess_current_preserves_state() {
        let mut state = GolgiCell::new();
        let before = snapshot(&state);

        assert_eq!(state.step(1.0e8), 0);

        assert_eq!(snapshot(&state), before);
    }

    #[test]
    fn test_golgi_cell_all_currents_bounded_and_calcium_active() {
        let mut state = GolgiCell::new();
        let baseline_ca = state.ca;
        let spikes: i32 = (0..2000).map(|_| state.step(10.0)).sum();

        assert!(spikes > 0);
        assert!(state.ca > baseline_ca);
        assert!(validate_golgi_cell(&state));
    }
}
