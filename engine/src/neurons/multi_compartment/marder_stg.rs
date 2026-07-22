// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Marder stomatogastric-ganglion neuron model

//! Marder stomatogastric-ganglion neuron model.

/// Marder STG — stomatogastric ganglion LP neuron with 7 currents.
#[derive(Clone, Debug)]
pub struct MarderSTGNeuron {
    pub v: f64,
    pub m_na: f64,
    pub h_na: f64,
    pub m_cat: f64,
    pub h_cat: f64,
    pub m_cas: f64,
    pub h_cas: f64,
    pub m_a: f64,
    pub h_a: f64,
    pub m_kca: f64,
    pub m_kd: f64,
    pub m_h: f64,
    pub ca: f64,
    pub cm: f64,
    pub g_na: f64,
    pub g_cat: f64,
    pub g_cas: f64,
    pub g_a: f64,
    pub g_kca: f64,
    pub g_kd: f64,
    pub g_h: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_h: f64,
    pub e_l: f64,
    pub ca_out: f64,
    pub ca_rest: f64,
    pub tau_ca: f64,
    pub f_ca: f64,
    pub celsius: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

// Boltzmann steady state 1/(1+exp((vh-v)/s)).
fn stg_sig(v: f64, vh: f64, s: f64) -> f64 {
    1.0 / (1.0 + stg_exp((vh - v) / s))
}

// Overflow-safe exp (argument clamped to [-700, 700]) — mirrors the Python model.
fn stg_exp(x: f64) -> f64 {
    x.clamp(-700.0, 700.0).exp()
}

impl MarderSTGNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            m_na: 0.0,
            h_na: 1.0,
            m_cat: 0.0,
            h_cat: 1.0,
            m_cas: 0.0,
            h_cas: 1.0,
            m_a: 0.0,
            h_a: 1.0,
            m_kca: 0.0,
            m_kd: 0.0,
            m_h: 0.0,
            ca: 0.05,
            cm: 1.0,
            g_na: 200.0,
            g_cat: 2.5,
            g_cas: 4.0,
            g_a: 50.0,
            g_kca: 25.0,
            g_kd: 75.0,
            g_h: 0.01,
            g_l: 0.01,
            e_na: 50.0,
            e_k: -80.0,
            e_h: -20.0,
            e_l: -50.0,
            ca_out: 3000.0,
            ca_rest: 0.05,
            tau_ca: 20.0,
            f_ca: 0.94,
            celsius: 10.0,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }

    fn nernst_e_ca(&self, ca: f64) -> f64 {
        let rt_zf = 1000.0 * 8.314462618 * (self.celsius + 273.15) / (2.0 * 96485.33212);
        rt_zf * (self.ca_out / ca.max(1e-9)).ln()
    }

    // d/dt of (v, m_na, h_na, m_cat, h_cat, m_cas, h_cas, m_a, h_a, m_kca, m_kd, m_h, ca).
    fn derivatives(&self, st: &[f64; 13], current: f64) -> [f64; 13] {
        let v = st[0];
        let (m_na, h_na) = (st[1], st[2]);
        let (m_cat, h_cat) = (st[3], st[4]);
        let (m_cas, h_cas) = (st[5], st[6]);
        let (m_a, h_a) = (st[7], st[8]);
        let (m_kca, m_kd, m_h, ca) = (st[9], st[10], st[11], st[12]);

        let tau_m_na = 1.32 - 1.26 / (1.0 + stg_exp(-(v + 120.0) / 25.0));
        let tau_h_na = (0.67 / (1.0 + stg_exp(-(v + 62.9) / 10.0)))
            * (1.5 + 1.0 / (1.0 + stg_exp((v + 34.9) / 3.6)));
        let tau_m_cat = 21.7 - 21.3 / (1.0 + stg_exp(-(v + 68.1) / 20.5));
        let tau_h_cat = 105.0 - 89.8 / (1.0 + stg_exp(-(v + 55.0) / 16.9));
        let tau_m_cas = 1.4 + 7.0 / (stg_exp((v + 27.0) / 10.0) + stg_exp(-(v + 70.0) / 13.0));
        let tau_h_cas = 60.0 + 150.0 / (stg_exp((v + 55.0) / 9.0) + stg_exp(-(v + 65.0) / 16.0));
        let tau_m_a = 11.6 - 10.4 / (1.0 + stg_exp(-(v + 32.9) / 15.2));
        let tau_h_a = 38.6 - 29.2 / (1.0 + stg_exp(-(v + 38.9) / 26.5));
        let tau_m_kca = 90.3 - 75.1 / (1.0 + stg_exp(-(v + 46.0) / 22.7));
        let tau_m_kd = 7.2 - 6.4 / (1.0 + stg_exp(-(v + 28.3) / 19.2));
        let tau_m_h = 272.0 + 1499.0 / (1.0 + stg_exp(-(v + 42.2) / 8.73));

        let m_kca_inf = (ca / (ca + 3.0)) * stg_sig(v, -28.3, 12.6);
        let e_ca = self.nernst_e_ca(ca);
        let i_na = self.g_na * m_na.powi(3) * h_na * (v - self.e_na);
        let i_cat = self.g_cat * m_cat.powi(3) * h_cat * (v - e_ca);
        let i_cas = self.g_cas * m_cas.powi(3) * h_cas * (v - e_ca);
        let i_a = self.g_a * m_a.powi(3) * h_a * (v - self.e_k);
        let i_kca = self.g_kca * m_kca.powi(4) * (v - self.e_k);
        let i_kd = self.g_kd * m_kd.powi(4) * (v - self.e_k);
        let i_h = self.g_h * m_h * (v - self.e_h);
        let i_l = self.g_l * (v - self.e_l);

        let dv = (current - i_na - i_cat - i_cas - i_a - i_kca - i_kd - i_h - i_l) / self.cm;
        let dca = (-self.f_ca * (i_cat + i_cas) - (ca - self.ca_rest)) / self.tau_ca;
        [
            dv,
            (stg_sig(v, -25.5, 5.29) - m_na) / tau_m_na,
            (stg_sig(v, -48.9, -5.18) - h_na) / tau_h_na,
            (stg_sig(v, -27.1, 7.2) - m_cat) / tau_m_cat,
            (stg_sig(v, -32.1, -5.5) - h_cat) / tau_h_cat,
            (stg_sig(v, -33.0, 8.1) - m_cas) / tau_m_cas,
            (stg_sig(v, -60.0, -6.2) - h_cas) / tau_h_cas,
            (stg_sig(v, -27.2, 8.7) - m_a) / tau_m_a,
            (stg_sig(v, -56.9, -4.9) - h_a) / tau_h_a,
            (m_kca_inf - m_kca) / tau_m_kca,
            (stg_sig(v, -12.3, 11.8) - m_kd) / tau_m_kd,
            (stg_sig(v, -70.0, -6.0) - m_h) / tau_m_h,
            dca,
        ]
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let y = [
            self.v, self.m_na, self.h_na, self.m_cat, self.h_cat, self.m_cas, self.h_cas, self.m_a,
            self.h_a, self.m_kca, self.m_kd, self.m_h, self.ca,
        ];
        let dt = self.dt;
        let k1 = self.derivatives(&y, current);
        let mut y2 = y;
        let mut y3 = y;
        let mut y4 = y;
        for i in 0..13 {
            y2[i] = y[i] + (dt / 2.0) * k1[i];
        }
        let k2 = self.derivatives(&y2, current);
        for i in 0..13 {
            y3[i] = y[i] + (dt / 2.0) * k2[i];
        }
        let k3 = self.derivatives(&y3, current);
        for i in 0..13 {
            y4[i] = y[i] + dt * k3[i];
        }
        let k4 = self.derivatives(&y4, current);
        let mut nxt = [0.0_f64; 13];
        for i in 0..13 {
            nxt[i] = y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        // Fail closed: an out-of-regime drive/timestep can drive the stiff RK4
        // step non-finite; keep the last finite state instead of propagating it.
        if !nxt.iter().all(|value| value.is_finite()) {
            return 0;
        }
        self.v = nxt[0];
        self.m_na = nxt[1].clamp(0.0, 1.0);
        self.h_na = nxt[2].clamp(0.0, 1.0);
        self.m_cat = nxt[3].clamp(0.0, 1.0);
        self.h_cat = nxt[4].clamp(0.0, 1.0);
        self.m_cas = nxt[5].clamp(0.0, 1.0);
        self.h_cas = nxt[6].clamp(0.0, 1.0);
        self.m_a = nxt[7].clamp(0.0, 1.0);
        self.h_a = nxt[8].clamp(0.0, 1.0);
        self.m_kca = nxt[9].clamp(0.0, 1.0);
        self.m_kd = nxt[10].clamp(0.0, 1.0);
        self.m_h = nxt[11].clamp(0.0, 1.0);
        self.ca = nxt[12].max(0.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -60.0;
        self.m_na = 0.0;
        self.h_na = 1.0;
        self.m_cat = 0.0;
        self.h_cat = 1.0;
        self.m_cas = 0.0;
        self.h_cas = 1.0;
        self.m_a = 0.0;
        self.h_a = 1.0;
        self.m_kca = 0.0;
        self.m_kd = 0.0;
        self.m_h = 0.0;
        self.ca = 0.05;
    }
}
impl Default for MarderSTGNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marder_fires() {
        let mut n = MarderSTGNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn marder_reset() {
        let mut n = MarderSTGNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-60.0)).abs() < 1e-10);
    }

    #[test]
    fn marder_bounded() {
        let mut n = MarderSTGNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn marder_nan_no_panic() {
        MarderSTGNeuron::new().step(f64::NAN);
    }
}
