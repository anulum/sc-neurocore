// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Multi-compartment neuron models

//! Multi-compartment neuron models.

#[allow(unused_imports)]
use super::biophysical::safe_rate;

/// Pinsky-Rinzel 1994 — 2-compartment pyramidal cell.
#[derive(Clone, Debug)]
pub struct PinskyRinzelNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub h: f64,
    pub n: f64,
    pub s: f64,
    pub c: f64,
    pub q: f64,
    pub ca: f64,
    pub cm: f64,
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
            v_s: -60.0,
            v_d: -60.0,
            h: 0.999,
            n: 0.001,
            s: 0.009,
            c: 0.007,
            q: 0.01,
            ca: 0.2,
            cm: 3.0,
            gc: 2.1,
            p: 0.5,
            g_na: 30.0,
            g_kdr: 15.0,
            g_ca: 10.0,
            g_kahp: 0.8,
            g_kc: 15.0,
            g_l: 0.1,
            e_na: 60.0,
            e_k: -75.0,
            e_ca: 80.0,
            e_l: -60.0,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }

    /// Time derivatives of the eight-state vector `(v_s, v_d, h, n, s, c, q, ca)`.
    fn derivatives(&self, st: &[f64; 8], i_s: f64, i_d: f64) -> [f64; 8] {
        let (v_s, v_d, h, n, s, c, q, ca) =
            (st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7]);
        let am = safe_rate(0.32, 46.9, v_s, 4.0, 0.32 * 4.0);
        let bm = pr_exprel_plus(0.28, v_s + 19.9, 5.0);
        let m_inf = if am + bm > 0.0 { am / (am + bm) } else { 0.0 };
        let ah = 0.128 * (-(v_s + 43.0) / 18.0).exp();
        let bh = 4.0 / (1.0 + (-(v_s + 20.0) / 5.0).exp());
        let an = safe_rate(0.016, 24.9, v_s, 5.0, 0.016 * 5.0);
        let bn = 0.25 * (-1.0 - 0.025 * v_s).exp();
        let a_s = 1.6 / (1.0 + (-0.072 * (v_d - 5.0)).exp());
        let b_s = pr_exprel_plus(0.02, v_d + 8.9, 5.0);
        let (ac, bc) = if v_d <= -10.0 {
            let ac = ((v_d + 50.0) / 11.0 - (v_d + 53.5) / 27.0).exp() / 18.975;
            (ac, 2.0 * ((-53.5 - v_d) / 27.0).exp() - ac)
        } else {
            (2.0 * ((-53.5 - v_d) / 27.0).exp(), 0.0)
        };
        let aq = (0.00002 * ca).min(0.01);
        let bq = 0.001;
        let chi = (ca / 250.0).min(1.0);

        let i_na = self.g_na * m_inf.powi(2) * h * (v_s - self.e_na);
        let i_kdr = self.g_kdr * n * (v_s - self.e_k);
        let i_ls = self.g_l * (v_s - self.e_l);
        let i_ca = self.g_ca * s.powi(2) * (v_d - self.e_ca);
        let i_kahp = self.g_kahp * q * (v_d - self.e_k);
        let i_kc = self.g_kc * c * chi * (v_d - self.e_k);
        let i_ld = self.g_l * (v_d - self.e_l);
        let i_coupling = self.gc * (v_d - v_s);

        let dv_s = (-i_ls - i_na - i_kdr + i_coupling / self.p + i_s / self.p) / self.cm;
        let dv_d = (-i_ld - i_ca - i_kahp - i_kc - i_coupling / (1.0 - self.p)
            + i_d / (1.0 - self.p))
            / self.cm;
        [
            dv_s,
            dv_d,
            ah * (1.0 - h) - bh * h,
            an * (1.0 - n) - bn * n,
            a_s * (1.0 - s) - b_s * s,
            ac * (1.0 - c) - bc * c,
            aq * (1.0 - q) - bq * q,
            -0.13 * i_ca - 0.075 * ca,
        ]
    }

    pub fn step(&mut self, current_soma: f64, current_dend: f64) -> i32 {
        let v_prev = self.v_s;
        let y = [
            self.v_s, self.v_d, self.h, self.n, self.s, self.c, self.q, self.ca,
        ];
        let dt = self.dt;
        let k1 = self.derivatives(&y, current_soma, current_dend);
        let mut y2 = y;
        for i in 0..8 {
            y2[i] = y[i] + (dt / 2.0) * k1[i];
        }
        let k2 = self.derivatives(&y2, current_soma, current_dend);
        let mut y3 = y;
        for i in 0..8 {
            y3[i] = y[i] + (dt / 2.0) * k2[i];
        }
        let k3 = self.derivatives(&y3, current_soma, current_dend);
        let mut y4 = y;
        for i in 0..8 {
            y4[i] = y[i] + dt * k3[i];
        }
        let k4 = self.derivatives(&y4, current_soma, current_dend);
        let mut nxt = [0.0_f64; 8];
        for i in 0..8 {
            nxt[i] = y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        self.v_s = nxt[0];
        self.v_d = nxt[1];
        self.h = nxt[2].clamp(0.0, 1.0);
        self.n = nxt[3].clamp(0.0, 1.0);
        self.s = nxt[4].clamp(0.0, 1.0);
        self.c = nxt[5].clamp(0.0, 1.0);
        self.q = nxt[6].clamp(0.0, 1.0);
        self.ca = nxt[7].max(0.0);
        if self.v_s >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -60.0;
        self.v_d = -60.0;
        self.h = 0.999;
        self.n = 0.001;
        self.s = 0.009;
        self.c = 0.007;
        self.q = 0.01;
        self.ca = 0.2;
    }
}

/// Traub deactivation rate `a*dv / (exp(dv/k) - 1)` with removable limit `a*k`.
fn pr_exprel_plus(a: f64, dv: f64, k: f64) -> f64 {
    if dv.abs() < 1e-6 {
        a * k
    } else {
        a * dv / ((dv / k).exp() - 1.0)
    }
}
impl Default for PinskyRinzelNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Hay et al. 2011 — Layer 5 thick-tufted pyramidal (3-compartment reduced).
#[derive(Clone, Debug)]
pub struct HayL5PyramidalNeuron {
    pub v_s: f64,
    pub v_t: f64,
    pub v_a: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub m_ca: f64,
    pub h_ca: f64,
    pub m_ih: f64,
    pub ca_a: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l_s: f64,
    pub g_ca_t: f64,
    pub g_ih: f64,
    pub g_l_t: f64,
    pub g_ca_a: f64,
    pub g_kca: f64,
    pub g_l_a: f64,
    pub g_st: f64,
    pub g_ta: f64,
    pub p_s: f64,
    pub p_t: f64,
    pub p_a: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_ih: f64,
    pub e_l: f64,
    pub ca_decay: f64,
    pub f_ca: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HayL5PyramidalNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -75.0,
            v_t: -75.0,
            v_a: -75.0,
            h_na: 0.9,
            n_k: 0.1,
            m_ca: 0.0,
            h_ca: 1.0,
            m_ih: 0.0,
            ca_a: 0.0001,
            g_na: 300.0,
            g_k: 40.0,
            g_l_s: 0.03,
            g_ca_t: 2.0,
            g_ih: 0.02,
            g_l_t: 0.03,
            g_ca_a: 1.5,
            g_kca: 2.5,
            g_l_a: 0.03,
            g_st: 1.5,
            g_ta: 0.8,
            p_s: 0.15,
            p_t: 0.25,
            p_a: 0.60,
            e_na: 50.0,
            e_k: -85.0,
            e_ca: 140.0,
            e_ih: -45.0,
            e_l: -75.0,
            ca_decay: 200.0,
            f_ca: 0.0002,
            c_m: 1.0,
            dt: 0.025,
            v_threshold: -30.0,
        }
    }
    fn valid(&self) -> bool {
        self.v_s.is_finite()
            && self.h_na.is_finite()
            && self.n_k.is_finite()
            && self.v_t.is_finite()
            && self.m_ca.is_finite()
            && self.h_ca.is_finite()
            && self.m_ih.is_finite()
            && self.v_a.is_finite()
            && self.ca_a.is_finite()
            && self.ca_a >= 0.0
            && self.g_na.is_finite()
            && self.g_na >= 0.0
            && self.g_k.is_finite()
            && self.g_k >= 0.0
            && self.g_l_s.is_finite()
            && self.g_l_s >= 0.0
            && self.g_ca_t.is_finite()
            && self.g_ca_t >= 0.0
            && self.g_ih.is_finite()
            && self.g_ih >= 0.0
            && self.g_l_t.is_finite()
            && self.g_l_t >= 0.0
            && self.g_ca_a.is_finite()
            && self.g_ca_a >= 0.0
            && self.g_kca.is_finite()
            && self.g_kca >= 0.0
            && self.g_l_a.is_finite()
            && self.g_l_a >= 0.0
            && self.g_st.is_finite()
            && self.g_st >= 0.0
            && self.g_ta.is_finite()
            && self.g_ta >= 0.0
            && self.p_s.is_finite()
            && self.p_s > 0.0
            && self.p_t.is_finite()
            && self.p_t > 0.0
            && self.p_a.is_finite()
            && self.p_a > 0.0
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_ca.is_finite()
            && self.e_ih.is_finite()
            && self.e_l.is_finite()
            && self.ca_decay.is_finite()
            && self.ca_decay > 0.0
            && self.f_ca.is_finite()
            && self.f_ca >= 0.0
            && self.c_m.is_finite()
            && self.c_m > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }

    fn derivatives(&self, s: [f64; 9], current_soma: f64, current_tuft: f64) -> [f64; 9] {
        let v_s = s[0];
        let h_na = s[1];
        let n_k = s[2];
        let v_t = s[3];
        let m_ca = s[4];
        let h_ca = s[5];
        let m_ih = s[6];
        let v_a = s[7];
        let ca_a = s[8].max(0.0);

        let m_na_inf = 1.0 / (1.0 + (-(v_s + 38.0) / 7.0).exp());
        let h_na_inf = 1.0 / (1.0 + ((v_s + 65.0) / 6.0).exp());
        let n_k_inf = 1.0 / (1.0 + (-(v_s + 25.0) / 12.0).exp());
        let tau_h = 0.5 + 14.0 / (1.0 + ((v_s + 35.0) / 10.0).exp());
        let tau_n = 1.0 + 5.0 / (1.0 + ((v_s + 30.0) / 10.0).exp());
        let i_na = self.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v_s - self.e_na);
        let i_k = self.g_k * n_k * n_k * n_k * n_k * (v_s - self.e_k);
        let i_l_s = self.g_l_s * (v_s - self.e_l);
        let i_st = self.g_st * (v_s - v_t) / self.p_s;

        let m_ca_inf = 1.0 / (1.0 + (-(v_t + 27.0) / 7.0).exp());
        let h_ca_inf = 1.0 / (1.0 + ((v_t + 52.0) / 5.0).exp());
        let m_ih_inf = 1.0 / (1.0 + ((v_t + 75.0) / 5.5).exp());
        let i_ca_t = self.g_ca_t * m_ca * m_ca * h_ca * (v_t - self.e_ca);
        let i_ih = self.g_ih * m_ih * (v_t - self.e_ih);
        let i_l_t = self.g_l_t * (v_t - self.e_l);
        let i_ts = self.g_st * (v_t - v_s) / self.p_t;
        let i_ta = self.g_ta * (v_t - v_a) / self.p_t;

        let m_ca_a_inf = 1.0 / (1.0 + (-(v_a + 30.0) / 5.0).exp());
        let kca_act = ca_a / (ca_a + 0.001);
        let i_ca_a = self.g_ca_a * m_ca_a_inf * m_ca_a_inf * (v_a - self.e_ca);
        let i_kca = self.g_kca * kca_act * (v_a - self.e_k);
        let i_l_a = self.g_l_a * (v_a - self.e_l);
        let i_at = self.g_ta * (v_a - v_t) / self.p_a;

        [
            (-i_na - i_k - i_l_s - i_st + current_soma / self.p_s) / self.c_m,
            (h_na_inf - h_na) / tau_h,
            (n_k_inf - n_k) / tau_n,
            (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / self.c_m,
            m_ca_inf - m_ca,
            (h_ca_inf - h_ca) / 20.0,
            (m_ih_inf - m_ih) / 50.0,
            (-i_ca_a - i_kca - i_l_a - i_at + current_tuft / self.p_a) / self.c_m,
            -self.f_ca * i_ca_a - ca_a / self.ca_decay,
        ]
    }

    fn rk4_substep(&self, s: [f64; 9], current_soma: f64, current_tuft: f64) -> [f64; 9] {
        let dt = self.dt;
        let k1 = self.derivatives(s, current_soma, current_tuft);
        let mut s2 = [0.0; 9];
        let mut s3 = [0.0; 9];
        let mut s4 = [0.0; 9];
        for i in 0..9 {
            s2[i] = s[i] + 0.5 * dt * k1[i];
        }
        let k2 = self.derivatives(s2, current_soma, current_tuft);
        for i in 0..9 {
            s3[i] = s[i] + 0.5 * dt * k2[i];
        }
        let k3 = self.derivatives(s3, current_soma, current_tuft);
        for i in 0..9 {
            s4[i] = s[i] + dt * k3[i];
        }
        let k4 = self.derivatives(s4, current_soma, current_tuft);
        let mut next = [0.0; 9];
        for i in 0..9 {
            next[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        next[8] = next[8].max(0.0);
        next
    }

    pub fn step(&mut self, current_soma: f64, current_tuft: f64) -> i32 {
        if !current_soma.is_finite() || !current_tuft.is_finite() || !self.valid() {
            return 0;
        }
        let v_s_prev = self.v_s;
        let mut state = [
            self.v_s, self.h_na, self.n_k, self.v_t, self.m_ca, self.h_ca, self.m_ih, self.v_a,
            self.ca_a,
        ];
        for _ in 0..4 {
            state = self.rk4_substep(state, current_soma, current_tuft);
            if !state.iter().all(|value| value.is_finite()) {
                return 0;
            }
        }
        self.v_s = state[0];
        self.h_na = state[1];
        self.n_k = state[2];
        self.v_t = state[3];
        self.m_ca = state[4];
        self.h_ca = state[5];
        self.m_ih = state[6];
        self.v_a = state[7];
        self.ca_a = state[8];
        if self.v_s >= self.v_threshold && v_s_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -75.0;
        self.v_t = -75.0;
        self.v_a = -75.0;
        self.h_na = 0.9;
        self.n_k = 0.1;
        self.m_ca = 0.0;
        self.h_ca = 1.0;
        self.m_ih = 0.0;
        self.ca_a = 0.0001;
    }
}
impl Default for HayL5PyramidalNeuron {
    fn default() -> Self {
        Self::new()
    }
}

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

/// Rall cable — N-compartment passive dendrite model. Rall 1964.
#[derive(Clone, Debug)]
pub struct RallCableNeuron {
    pub v: Vec<f64>,
    pub n_comp: usize,
    pub tau_m: f64,
    pub v_rest: f64,
    pub g_ratio: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl RallCableNeuron {
    pub fn new(n_comp: usize) -> Self {
        let count = n_comp.max(1);
        Self {
            v: vec![-65.0; count],
            n_comp: count,
            tau_m: 20.0,
            v_rest: -65.0,
            g_ratio: 0.5,
            v_threshold: -50.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let Some(mut candidate) = self.candidate(current) else {
            return -1;
        };
        let previous_soma = self.v[0];
        if candidate[0] >= self.v_threshold && previous_soma < self.v_threshold {
            candidate[0] = self.v_reset;
            self.v = candidate;
            1
        } else {
            self.v = candidate;
            0
        }
    }
    pub fn reset(&mut self) {
        self.v.fill(self.v_rest);
    }

    fn valid(&self) -> bool {
        self.n_comp >= 1
            && self.v.len() == self.n_comp
            && self.tau_m.is_finite()
            && self.tau_m > 0.0
            && self.v_rest.is_finite()
            && self.g_ratio.is_finite()
            && self.g_ratio >= 0.0
            && self.v_threshold.is_finite()
            && self.v_reset.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v.iter().all(|value| value.is_finite())
    }

    fn candidate(&self, current: f64) -> Option<Vec<f64>> {
        if !self.valid() || !current.is_finite() {
            return None;
        }
        let alpha = self.dt / self.tau_m;
        let offdiag = -alpha * self.g_ratio;
        let mut diagonal = vec![1.0 + alpha + 2.0 * alpha * self.g_ratio; self.n_comp];
        if self.n_comp == 1 {
            diagonal[0] = 1.0 + alpha;
        } else {
            diagonal[0] = 1.0 + alpha + alpha * self.g_ratio;
            diagonal[self.n_comp - 1] = 1.0 + alpha + alpha * self.g_ratio;
        }
        let lower = vec![offdiag; self.n_comp.saturating_sub(1)];
        let upper = vec![offdiag; self.n_comp.saturating_sub(1)];
        let mut rhs: Vec<f64> = self.v.iter().map(|value| value - self.v_rest).collect();
        rhs[self.n_comp - 1] += alpha * current;
        let mut solved = solve_rall_tridiagonal(&lower, &diagonal, &upper, &rhs)?;
        for value in &mut solved {
            *value += self.v_rest;
        }
        Some(solved)
    }
}

fn solve_rall_tridiagonal(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let n = diagonal.len();
    if n == 0
        || rhs.len() != n
        || lower.len() != n.saturating_sub(1)
        || upper.len() != n.saturating_sub(1)
    {
        return None;
    }
    let mut c_prime = vec![0.0; n.saturating_sub(1)];
    let mut d_prime = vec![0.0; n];
    let mut pivot = diagonal[0];
    if !pivot.is_finite() || pivot == 0.0 {
        return None;
    }
    if n > 1 {
        c_prime[0] = upper[0] / pivot;
    }
    d_prime[0] = rhs[0] / pivot;
    for i in 1..n {
        pivot = diagonal[i] - lower[i - 1] * c_prime[i - 1];
        if !pivot.is_finite() || pivot == 0.0 {
            return None;
        }
        if i < n - 1 {
            c_prime[i] = upper[i] / pivot;
        }
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / pivot;
    }
    let mut solution = vec![0.0; n];
    solution[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1];
    }
    solution
        .iter()
        .all(|value| value.is_finite())
        .then_some(solution)
}

/// Booth-Rinzel — 2-compartment motoneuron with bistability. Booth et al. 1997.
#[derive(Clone, Debug)]
pub struct BoothRinzelNeuron {
    pub vs: f64,
    pub vd: f64,
    pub h: f64,
    pub n: f64,
    pub q: f64,
    pub ca: f64,
    pub p: f64,
    pub gc: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_ca: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl BoothRinzelNeuron {
    pub fn new() -> Self {
        Self {
            vs: -65.0,
            vd: -65.0,
            h: 0.9,
            n: 0.0,
            q: 0.0,
            ca: 0.0,
            p: 0.5,
            gc: 0.1,
            g_na: 120.0,
            g_k: 20.0,
            g_ca: 14.0,
            g_kca: 5.0,
            g_l: 0.51,
            e_na: 55.0,
            e_k: -80.0,
            e_ca: 80.0,
            e_l: -60.0,
            dt: 0.025,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.vs;
        for _ in 0..4 {
            let m_inf = 1.0 / (1.0 + (-(self.vs + 35.0) / 7.8).exp());
            let h_inf = 1.0 / (1.0 + ((self.vs + 55.0) / 7.0).exp());
            let n_inf = 1.0 / (1.0 + (-(self.vs + 28.0) / 15.0).exp());
            let s_inf = 1.0 / (1.0 + (-(self.vd + 22.0) / 5.0).exp());
            let q_inf = 1.0 / (1.0 + (-(self.vd + 35.0) / 2.0).exp());
            let tau_h = (30.0
                / (((self.vs + 50.0) / 15.0).exp() + ((-(self.vs + 50.0)) / 16.0).exp() + 1e-12))
                .max(0.01);
            let tau_n = (7.0
                / (((self.vs + 40.0) / 40.0).exp() + ((-(self.vs + 40.0)) / 50.0).exp() + 1e-12))
                .max(0.01);
            self.h = (self.h + (h_inf - self.h) / tau_h * self.dt).clamp(0.0, 1.0);
            self.n = (self.n + (n_inf - self.n) / tau_n * self.dt).clamp(0.0, 1.0);
            self.q = (self.q + (q_inf - self.q) / 400.0 * self.dt).clamp(0.0, 1.0);
            let chi = (self.ca / 250.0).min(1.0);
            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.vs - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.vs - self.e_k);
            let i_ls = self.g_l * (self.vs - self.e_l);
            let i_sd = (self.gc / self.p) * (self.vs - self.vd);
            let i_ca = self.g_ca * s_inf.powi(2) * (self.vd - self.e_ca);
            let i_kca = self.g_kca * chi * (self.vd - self.e_k);
            let i_ld = self.g_l * (self.vd - self.e_l);
            let i_ds = (self.gc / (1.0 - self.p)) * (self.vd - self.vs);
            self.vs = (self.vs + (-i_na - i_k - i_ls - i_sd + current / self.p) * self.dt)
                .clamp(-200.0, 100.0);
            self.vd = (self.vd + (-i_ca - i_kca - i_ld - i_ds) * self.dt).clamp(-200.0, 100.0);
            self.ca = (self.ca + (0.0025 * (-0.009 * i_ca) - 0.18 * self.ca) * self.dt).max(0.0);
        }
        if self.vs >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.vs = -65.0;
        self.vd = -65.0;
        self.h = 0.9;
        self.n = 0.0;
        self.q = 0.0;
        self.ca = 0.0;
    }
}
impl Default for BoothRinzelNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Dendrify — two-compartment with active dendritic spike (NMDA-like plateau).
#[derive(Clone, Debug)]
pub struct DendrifyNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub d_active: bool,
    pub d_timer: f64,
    pub tau_s: f64,
    pub tau_d: f64,
    pub g_c: f64,
    pub d_threshold: f64,
    pub d_amplitude: f64,
    pub d_duration: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl DendrifyNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -65.0,
            v_d: -65.0,
            d_active: false,
            d_timer: 0.0,
            tau_s: 10.0,
            tau_d: 20.0,
            g_c: 0.8,
            d_threshold: -35.0,
            d_amplitude: 30.0,
            d_duration: 10.0,
            v_rest: -65.0,
            v_threshold: -50.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let d_input = if self.d_active { self.d_amplitude } else { 0.0 };
        self.v_d += (-(self.v_d - self.v_rest) + current + d_input
            - self.g_c * (self.v_d - self.v_s))
            / self.tau_d
            * self.dt;
        self.v_s +=
            (-(self.v_s - self.v_rest) + self.g_c * (self.v_d - self.v_s)) / self.tau_s * self.dt;
        if self.d_active {
            self.d_timer -= self.dt;
            if self.d_timer <= 0.0 {
                self.d_active = false;
            }
        } else if self.v_d >= self.d_threshold {
            self.d_active = true;
            self.d_timer = self.d_duration;
        }
        if self.v_s >= self.v_threshold {
            self.v_s = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -65.0;
        self.v_d = -65.0;
        self.d_active = false;
        self.d_timer = 0.0;
    }
}
impl Default for DendrifyNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Two-compartment LIF — soma + dendrite with history-dependent coupling.
#[derive(Clone, Debug)]
pub struct TwoCompartmentLIFNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta: f64,
    pub tau_s: f64,
    pub tau_d: f64,
    pub kappa: f64,
    pub dt: f64,
}

impl TwoCompartmentLIFNeuron {
    pub fn new() -> Self {
        Self {
            v_s: 0.0,
            v_d: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            theta: 1.0,
            tau_s: 2.0,
            tau_d: 20.0,
            kappa: 0.5,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, i_soma: f64, i_dend: f64) -> i32 {
        let alpha_s = (-self.dt / self.tau_s).exp();
        let alpha_d = (-self.dt / self.tau_d).exp();
        self.v_d = alpha_d * self.v_d + i_dend;
        self.v_s = alpha_s * self.v_s + i_soma + self.kappa * self.v_d;
        if self.v_s >= self.theta {
            self.v_s = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = self.v_rest;
        self.v_d = self.v_rest;
    }
}
impl Default for TwoCompartmentLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pr_fires() {
        let mut n = PinskyRinzelNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(5.0, 0.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn hay_fires() {
        let mut n = HayL5PyramidalNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(20.0, 0.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn marder_fires() {
        let mut n = MarderSTGNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn rall_fires() {
        let mut n = RallCableNeuron::new(2);
        n.g_ratio = 5.0;
        let t: i32 = (0..5000).map(|_| n.step(500.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn booth_fires() {
        let mut n = BoothRinzelNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn dendrify_fires() {
        let mut n = DendrifyNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(50.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn tc_lif_fires() {
        let mut n = TwoCompartmentLIFNeuron::new();
        let t: i32 = (0..100).map(|_| n.step(0.5, 0.3)).sum();
        assert!(t > 0);
    }

    // ── Multi-angle tests for multi-compartment models ──

    // -- PinskyRinzel --
    #[test]
    fn pr_reset() {
        let mut n = PinskyRinzelNeuron::new();
        for _ in 0..100 {
            n.step(5.0, 0.0);
        }
        n.reset();
        assert!((n.v_s - (-60.0)).abs() < 1e-10);
        assert!((n.v_d - (-60.0)).abs() < 1e-10);
    }
    #[test]
    fn pr_bounded() {
        let mut n = PinskyRinzelNeuron::new();
        for _ in 0..5000 {
            n.step(50.0, 0.0);
        }
        assert!(n.v_s.is_finite());
    }
    #[test]
    fn pr_dendritic_input() {
        let mut n = PinskyRinzelNeuron::new();
        let _t: i32 = (0..5000).map(|_| n.step(0.0, 5.0)).sum();
        // Dendritic input should also be able to drive spiking
        assert!(n.v_d.is_finite());
    }
    #[test]
    fn pr_nan_no_panic() {
        PinskyRinzelNeuron::new().step(f64::NAN, 0.0);
    }

    // -- HayL5 --
    #[test]
    fn hay_reset() {
        let mut n = HayL5PyramidalNeuron::new();
        for _ in 0..100 {
            n.step(20.0, 0.0);
        }
        n.reset();
        assert!((n.v_s - (-75.0)).abs() < 1e-10);
    }
    #[test]
    fn hay_bounded() {
        let mut n = HayL5PyramidalNeuron::new();
        for _ in 0..500 {
            n.step(100.0, 0.0);
        }
        assert!(n.v_s.is_finite());
    }
    #[test]
    fn hay_nan_no_panic() {
        HayL5PyramidalNeuron::new().step(f64::NAN, 0.0);
    }
    #[test]
    fn hay_rk4_somatic_anchor() {
        let mut n = HayL5PyramidalNeuron::new();
        let spikes: i32 = (0..20_000).map(|_| n.step(10.0, 0.0)).sum();
        assert_eq!(spikes, 1);
        assert!(n.ca_a >= 0.0);
    }
    #[test]
    fn hay_rk4_dual_input_anchor() {
        let mut n = HayL5PyramidalNeuron::new();
        let spikes: i32 = (0..20_000).map(|_| n.step(5.0, 5.0)).sum();
        assert_eq!(spikes, 4);
    }
    #[test]
    fn hay_invalid_input_preserves_state() {
        let mut n = HayL5PyramidalNeuron::new();
        for _ in 0..10 {
            n.step(10.0, 0.0);
        }
        let old = [
            n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a,
        ];
        assert_eq!(n.step(f64::INFINITY, 0.0), 0);
        assert_eq!(
            [n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a],
            old
        );
    }

    // -- MarderSTG --
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

    // -- RallCable --
    #[test]
    fn rall_reset() {
        let mut n = RallCableNeuron::new(5);
        for _ in 0..100 {
            n.step(50.0);
        }
        n.reset();
        assert!(n.v.iter().all(|&x| (x - n.v_rest).abs() < 1e-10));
    }
    #[test]
    fn rall_bounded() {
        let mut n = RallCableNeuron::new(5);
        for _ in 0..1000 {
            n.step(500.0);
        }
        assert!(n.v.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn rall_implicit_step_reference() {
        let mut n = RallCableNeuron::new(3);
        assert_eq!(n.step(100.0), 0);
        assert!((n.v[0] - -64.99999695179709).abs() < 1e-12);
        assert!((n.v[1] - -64.99877157422763).abs() < 1e-12);
        assert!((n.v[2] - -64.50371903616434).abs() < 1e-12);
    }
    #[test]
    fn rall_nan_no_panic() {
        let mut n = RallCableNeuron::new(5);
        let before = n.v.clone();
        assert_eq!(n.step(f64::NAN), -1);
        assert_eq!(n.v, before);
    }

    // -- BoothRinzel --
    #[test]
    fn booth_reset() {
        let mut n = BoothRinzelNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.vs - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn booth_bounded() {
        let mut n = BoothRinzelNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.vs.is_finite());
    }
    #[test]
    fn booth_nan_no_panic() {
        BoothRinzelNeuron::new().step(f64::NAN);
    }

    // -- Dendrify --
    #[test]
    fn dendrify_reset() {
        let mut n = DendrifyNeuron::new();
        for _ in 0..100 {
            n.step(50.0);
        }
        n.reset();
        assert!((n.v_s - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn dendrify_bounded() {
        let mut n = DendrifyNeuron::new();
        for _ in 0..2000 {
            n.step(200.0);
        }
        assert!(n.v_s.is_finite());
    }
    #[test]
    fn dendrify_nan_no_panic() {
        DendrifyNeuron::new().step(f64::NAN);
    }

    // -- TwoCompartmentLIF --
    #[test]
    fn tc_lif_reset() {
        let mut n = TwoCompartmentLIFNeuron::new();
        for _ in 0..50 {
            n.step(0.5, 0.3);
        }
        n.reset();
    }
    #[test]
    fn tc_lif_bounded() {
        let mut n = TwoCompartmentLIFNeuron::new();
        for _ in 0..1000 {
            n.step(100.0, 100.0);
        }
        assert!(n.v_s.is_finite());
    }
    #[test]
    fn tc_lif_nan_no_panic() {
        TwoCompartmentLIFNeuron::new().step(f64::NAN, 0.0);
    }
}

/// Dendritic NMDA spike model.
///
/// Captures the non-linear voltage-dependent Mg²⁺ block of NMDA receptors
/// in dendritic branches. NMDA current has a sigmoidal voltage dependence:
///
///   I_NMDA = g_NMDA · B(V) · (V - E_NMDA)
///   B(V) = 1 / (1 + [Mg²⁺]/3.57 · exp(-0.062 · V))
///
/// This enables coincidence detection: the dendrite only passes current
/// when both presynaptic glutamate AND postsynaptic depolarisation are present.
///
/// Reference: Jahr & Stevens (1990), Schiller et al. (2000).
#[derive(Clone, Debug)]
pub struct DendriticNMDANeuron {
    pub v_soma: f64,
    pub v_dend: f64,
    pub g_nmda: f64,
    pub e_nmda: f64,
    pub mg_conc: f64,
    pub g_coupling: f64,
    pub tau_soma: f64,
    pub tau_dend: f64,
    pub theta: f64,
    pub dt: f64,
}

impl DendriticNMDANeuron {
    pub fn new() -> Self {
        Self {
            v_soma: -65.0,
            v_dend: -65.0,
            g_nmda: 1.5,
            e_nmda: 0.0,
            mg_conc: 1.0,
            g_coupling: 0.5,
            tau_soma: 20.0,
            tau_dend: 50.0,
            theta: -50.0,
            dt: 0.1,
        }
    }

    /// Mg²⁺ block factor (Jahr & Stevens 1990).
    fn mg_block(&self, v: f64) -> f64 {
        1.0 / (1.0 + (self.mg_conc / 3.57) * (-0.062 * v).exp())
    }

    fn valid(&self) -> bool {
        self.v_soma.is_finite()
            && self.v_dend.is_finite()
            && self.g_nmda.is_finite()
            && self.g_nmda >= 0.0
            && self.e_nmda.is_finite()
            && self.mg_conc.is_finite()
            && self.mg_conc >= 0.0
            && self.g_coupling.is_finite()
            && self.g_coupling >= 0.0
            && self.tau_soma.is_finite()
            && self.tau_soma > 0.0
            && self.tau_dend.is_finite()
            && self.tau_dend > 0.0
            && self.theta.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
    }

    fn derivatives(&self, v_soma: f64, v_dend: f64, i_soma: f64, glutamate: f64) -> (f64, f64) {
        let b = self.mg_block(v_dend);
        let i_nmda = self.g_nmda * glutamate * b * (v_dend - self.e_nmda);
        let dv_soma =
            (-v_soma - 65.0 + i_soma + self.g_coupling * (v_dend - v_soma)) / self.tau_soma;
        let dv_dend =
            (-v_dend - 65.0 + i_nmda + self.g_coupling * (v_soma - v_dend)) / self.tau_dend;
        (dv_soma, dv_dend)
    }

    fn rk4_substep(&self, v_soma: f64, v_dend: f64, i_soma: f64, glutamate: f64) -> (f64, f64) {
        let dt = self.dt;
        let (k1s, k1d) = self.derivatives(v_soma, v_dend, i_soma, glutamate);
        let (k2s, k2d) = self.derivatives(
            v_soma + 0.5 * dt * k1s,
            v_dend + 0.5 * dt * k1d,
            i_soma,
            glutamate,
        );
        let (k3s, k3d) = self.derivatives(
            v_soma + 0.5 * dt * k2s,
            v_dend + 0.5 * dt * k2d,
            i_soma,
            glutamate,
        );
        let (k4s, k4d) = self.derivatives(v_soma + dt * k3s, v_dend + dt * k3d, i_soma, glutamate);
        (
            v_soma + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0,
            v_dend + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0,
        )
    }

    /// Step with somatic input and dendritic glutamate.
    pub fn step(&mut self, i_soma: f64, glutamate: f64) -> i32 {
        if !i_soma.is_finite() || !glutamate.is_finite() || glutamate < 0.0 || !self.valid() {
            return 0;
        }
        let (next_v_soma, next_v_dend) =
            self.rk4_substep(self.v_soma, self.v_dend, i_soma, glutamate);
        if !next_v_soma.is_finite() || !next_v_dend.is_finite() {
            return 0;
        }
        self.v_dend = next_v_dend;
        if next_v_soma >= self.theta {
            self.v_soma = -65.0;
            1
        } else {
            self.v_soma = next_v_soma;
            0
        }
    }

    pub fn reset(&mut self) {
        self.v_soma = -65.0;
        self.v_dend = -65.0;
    }
}

impl Default for DendriticNMDANeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-compartment neuron (MCN) matching the Spiking-WM architecture.
///
/// Dual-dendrite model with basal and apical compartments. The apical dendrite
/// gates how strongly basal information influences the soma, enabling
/// nonlinear integration for long-term temporal memory in RL tasks. The engine
/// uses candidate-first RK4 over `(u, v_basal, v_apical)` so all compartments
/// are advanced from one consistent state before the reset is committed.
///
/// Exact equations from arXiv:2503.00713 (Spiking-WM, PNAS 2025):
///
///   τ_b dV_b/dt = -V_b + x_b                                  (basal)
///   τ_a dV_a/dt = -V_a + x_a                                  (apical)
///   τ   dU/dt   = -U + σ(V_a)·[g_B/g_L·(V_b - U) + W_s·I]   (soma)
///   S[t] = Θ(U[t] - V_th)                                     (spike)
///   U[t] ← U[t]·(1 - S[t])                                    (soft reset)
///
/// Default parameters from Table II: τ = τ_a = τ_b = 2.0, g_B/g_L = 1.0,
/// β = 1.0 (sigmoid steepness), V_th = 1.0.
///
/// Reference: Brain-Cog-Lab, arXiv:2503.00713, PNAS 2025.
#[derive(Clone, Debug)]
pub struct MulticompartmentMCNNeuron {
    /// Somatic membrane potential.
    pub u: f64,
    /// Basal dendrite potential.
    pub v_basal: f64,
    /// Apical dendrite potential.
    pub v_apical: f64,
    /// Soma time constant.
    pub tau: f64,
    /// Basal dendrite time constant.
    pub tau_b: f64,
    /// Apical dendrite time constant.
    pub tau_a: f64,
    /// Basal-to-soma conductance ratio (g_B/g_L).
    pub g_ratio: f64,
    /// Sigmoid steepness for apical gating.
    pub beta: f64,
    /// Spike threshold.
    pub v_th: f64,
    /// Time step.
    pub dt: f64,
}

impl MulticompartmentMCNNeuron {
    pub fn new() -> Self {
        Self {
            u: 0.0,
            v_basal: 0.0,
            v_apical: 0.0,
            tau: 2.0,
            tau_b: 2.0,
            tau_a: 2.0,
            g_ratio: 1.0,
            beta: 1.0,
            v_th: 1.0,
            dt: 1.0,
        }
    }

    /// Sigmoid gating function σ(x) = 1/(1 + exp(-βx)).
    fn sigma(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-self.beta * x).exp())
    }

    fn valid(&self) -> bool {
        self.tau.is_finite()
            && self.tau > 0.0
            && self.tau_b.is_finite()
            && self.tau_b > 0.0
            && self.tau_a.is_finite()
            && self.tau_a > 0.0
            && self.g_ratio.is_finite()
            && self.g_ratio >= 0.0
            && self.beta.is_finite()
            && self.beta > 0.0
            && self.v_th.is_finite()
            && self.v_th > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.u.is_finite()
            && self.v_basal.is_finite()
            && self.v_apical.is_finite()
    }

    fn derivatives(
        &self,
        u: f64,
        v_basal: f64,
        v_apical: f64,
        x_basal: f64,
        x_apical: f64,
        i_soma: f64,
    ) -> [f64; 3] {
        let gate = self.sigma(v_apical);
        let du = (-u + gate * (self.g_ratio * (v_basal - u) + i_soma)) / self.tau;
        let dv_basal = (-v_basal + x_basal) / self.tau_b;
        let dv_apical = (-v_apical + x_apical) / self.tau_a;
        [du, dv_basal, dv_apical]
    }

    fn rk4_substep(&self, state: [f64; 3], x_basal: f64, x_apical: f64, i_soma: f64) -> [f64; 3] {
        let dt = self.dt;
        let k1 = self.derivatives(state[0], state[1], state[2], x_basal, x_apical, i_soma);
        let k2 = self.derivatives(
            state[0] + 0.5 * dt * k1[0],
            state[1] + 0.5 * dt * k1[1],
            state[2] + 0.5 * dt * k1[2],
            x_basal,
            x_apical,
            i_soma,
        );
        let k3 = self.derivatives(
            state[0] + 0.5 * dt * k2[0],
            state[1] + 0.5 * dt * k2[1],
            state[2] + 0.5 * dt * k2[2],
            x_basal,
            x_apical,
            i_soma,
        );
        let k4 = self.derivatives(
            state[0] + dt * k3[0],
            state[1] + dt * k3[1],
            state[2] + dt * k3[2],
            x_basal,
            x_apical,
            i_soma,
        );
        [
            state[0] + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            state[1] + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
            state[2] + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        ]
    }

    fn threshold_reached(&self, candidate_u: f64) -> bool {
        let margin = 16.0 * f64::EPSILON * self.v_th.abs().max(1.0);
        candidate_u >= self.v_th || (candidate_u - self.v_th).abs() <= margin
    }

    /// Step with basal input (x_b), apical input (x_a), and direct somatic input.
    pub fn step_compartments(&mut self, x_basal: f64, x_apical: f64, i_soma: f64) -> i32 {
        if !x_basal.is_finite() || !x_apical.is_finite() || !i_soma.is_finite() || !self.valid() {
            return 0;
        }
        let next = self.rk4_substep(
            [self.u, self.v_basal, self.v_apical],
            x_basal,
            x_apical,
            i_soma,
        );
        if !next.iter().all(|value| value.is_finite()) {
            return 0;
        }
        let spike = self.threshold_reached(next[0]);
        self.u = if spike { 0.0 } else { next[0] };
        self.v_basal = next[1];
        self.v_apical = next[2];
        i32::from(spike)
    }

    /// Simple step: input goes to basal dendrite only.
    pub fn step(&mut self, current: f64) -> i32 {
        self.step_compartments(current, 0.0, 0.0)
    }

    pub fn reset(&mut self) {
        self.u = 0.0;
        self.v_basal = 0.0;
        self.v_apical = 0.0;
    }
}

impl Default for MulticompartmentMCNNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Astrocyte-LIF hybrid unit with calcium wave feedback.
///
/// Models the tripartite synapse: a glial astrocyte monitors extracellular
/// glutamate from a paired LIF neuron and provides slow homeostatic feedback
/// via calcium-dependent gliotransmitter release.
///
///   dCa/dt = -Ca/τ_ca + δ · S_pre(t)        (calcium rise on presynaptic spike)
///   I_glio = g_glio · H(Ca - Ca_thresh)      (gliotransmitter release)
///   dV/dt = -(V - E_L)/τ_m + I_ext + I_glio  (LIF with glial feedback)
///
/// Reference: Perea, Navarrete & Araque, "Tripartite synapses" (2009).
#[derive(Clone, Debug)]
pub struct AstrocyteLIFNeuron {
    pub v: f64,
    pub ca: f64,
    pub tau_m: f64,
    pub tau_ca: f64,
    pub e_l: f64,
    pub theta: f64,
    pub v_reset: f64,
    pub ca_delta: f64,
    pub ca_thresh: f64,
    pub g_glio: f64,
    pub dt: f64,
}

impl AstrocyteLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            ca: 0.0,
            tau_m: 20.0,
            tau_ca: 500.0,
            e_l: -65.0,
            theta: -50.0,
            v_reset: -65.0,
            ca_delta: 0.1,
            ca_thresh: 0.5,
            g_glio: 2.0,
            dt: 0.1,
        }
    }

    /// Step with external current and presynaptic spike indicator.
    pub fn step_with_pre(&mut self, i_ext: f64, pre_spike: bool) -> i32 {
        // Astrocyte calcium dynamics.
        let dca = -self.ca / self.tau_ca
            + if pre_spike {
                self.ca_delta / self.dt
            } else {
                0.0
            };
        self.ca += dca * self.dt;
        self.ca = self.ca.max(0.0);

        // Gliotransmitter release (Heaviside on calcium).
        let i_glio = if self.ca > self.ca_thresh {
            self.g_glio
        } else {
            0.0
        };

        // LIF membrane dynamics with glial feedback.
        let dv = (-(self.v - self.e_l) + i_ext + i_glio) / self.tau_m;
        self.v += dv * self.dt;

        if self.v >= self.theta {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    /// Simple step (no presynaptic spike).
    pub fn step(&mut self, current: f64) -> i32 {
        self.step_with_pre(current, false)
    }

    pub fn reset(&mut self) {
        self.v = self.e_l;
        self.ca = 0.0;
    }
}

impl Default for AstrocyteLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Tests for new multi-compartment / glial models ----

#[cfg(test)]
mod gap_mc_tests {
    use super::*;

    #[test]
    fn nmda_coincidence_detection() {
        let mut n = DendriticNMDANeuron::new();
        // Only soma input — dendrite contributes little.
        let mut spikes_soma_only = 0;
        for _ in 0..2000 {
            spikes_soma_only += n.step(8.0, 0.0);
        }
        n.reset();
        // Soma + glutamate — NMDA amplifies.
        let mut spikes_both = 0;
        for _ in 0..2000 {
            spikes_both += n.step(8.0, 1.0);
        }
        // Coincidence: both inputs together should fire more.
        assert!(
            spikes_both >= spikes_soma_only,
            "NMDA coincidence: both={spikes_both} must >= soma_only={spikes_soma_only}"
        );
    }

    #[test]
    fn nmda_mg_block_voltage_dependent() {
        let n = DendriticNMDANeuron::new();
        let b_hyper = n.mg_block(-80.0);
        let b_depol = n.mg_block(-20.0);
        assert!(
            b_depol > b_hyper,
            "Mg block must relieve at depolarised potentials: B(-20)={b_depol:.3} > B(-80)={b_hyper:.3}"
        );
    }

    #[test]
    fn nmda_zero_glutamate_no_nmda_current() {
        let mut n = DendriticNMDANeuron::new();
        let spikes: i32 = (0..500).map(|_| n.step(0.0, 0.0)).sum();
        assert_eq!(spikes, 0, "No input → no spikes");
    }

    #[test]
    fn nmda_rk4_cross_backend_anchor() {
        let mut n = DendriticNMDANeuron::new();
        let spikes: i32 = (0..20_000).map(|_| n.step(50.0, 0.5)).sum();
        assert_eq!(spikes, 253);
        assert!(n.v_soma.is_finite());
        assert!(n.v_dend.is_finite());
    }

    #[test]
    fn nmda_invalid_input_preserves_state() {
        let mut n = DendriticNMDANeuron::new();
        for _ in 0..10 {
            let _ = n.step(50.0, 0.5);
        }
        let old = (n.v_soma, n.v_dend);
        assert_eq!(n.step(f64::INFINITY, 0.5), 0);
        assert_eq!((n.v_soma, n.v_dend), old);
        assert_eq!(n.step(50.0, -1.0), 0);
        assert_eq!((n.v_soma, n.v_dend), old);
    }

    #[test]
    fn nmda_invalid_configuration_preserves_state() {
        let mut n = DendriticNMDANeuron::new();
        for _ in 0..10 {
            let _ = n.step(50.0, 0.5);
        }
        let old = (n.v_soma, n.v_dend);
        n.tau_dend = 0.0;
        assert_eq!(n.step(50.0, 0.5), 0);
        assert_eq!((n.v_soma, n.v_dend), old);
    }

    #[test]
    fn mcn_apical_gating() {
        // Without apical input, gate = σ(0) = 0.5, moderate drive.
        let mut n_no_apical = MulticompartmentMCNNeuron::new();
        let mut spikes_no = 0;
        for _ in 0..1000 {
            spikes_no += n_no_apical.step_compartments(2.5, 0.0, 0.0);
        }
        // With strong apical input, gate ≈ 1.0, full basal→soma coupling.
        let mut n_apical = MulticompartmentMCNNeuron::new();
        let mut spikes_yes = 0;
        for _ in 0..1000 {
            spikes_yes += n_apical.step_compartments(2.5, 5.0, 0.0);
        }
        assert!(
            spikes_yes >= spikes_no && spikes_yes > 0,
            "Apical gating should boost firing: apical={spikes_yes} >= none={spikes_no}"
        );
    }

    #[test]
    fn mcn_rk4_cross_backend_anchor() {
        let mut n = MulticompartmentMCNNeuron::new();
        let mut spikes = 0;
        for _ in 0..200_000 {
            spikes += n.step(3.2);
        }
        assert_eq!(spikes, 49_999);
    }

    #[test]
    fn mcn_threshold_boundary_accepts_one_ulp_roundoff() {
        let n = MulticompartmentMCNNeuron::new();
        let one_ulp_below = f64::from_bits(n.v_th.to_bits() - 1);
        assert!(n.threshold_reached(one_ulp_below));
        assert!(!n.threshold_reached(n.v_th - 1.0e-9));
    }

    #[test]
    fn mcn_invalid_input_preserves_state() {
        let mut n = MulticompartmentMCNNeuron::new();
        for _ in 0..5 {
            let _ = n.step(3.2);
        }
        let old = (n.u, n.v_basal, n.v_apical);
        assert_eq!(n.step(f64::INFINITY), 0);
        assert_eq!((n.u, n.v_basal, n.v_apical), old);
    }

    #[test]
    fn mcn_basal_dendrite_memory() {
        // τ_b = 2.0, dt = 1.0: V_b decays by factor (1 - dt/τ) = 0.5 per step.
        let mut n = MulticompartmentMCNNeuron::new();
        n.step_compartments(5.0, 0.0, 0.0);
        let v_after = n.v_basal;
        n.step_compartments(0.0, 0.0, 0.0);
        let v_decay = n.v_basal;
        assert!(
            v_decay.abs() > 0.1 * v_after.abs(),
            "Basal dendrite retains memory: {v_decay:.3} vs {v_after:.3}"
        );
    }

    #[test]
    fn mcn_reset_clears_all() {
        let mut n = MulticompartmentMCNNeuron::new();
        for _ in 0..50 {
            n.step(2.0);
        }
        n.reset();
        assert_eq!(n.u, 0.0);
        assert_eq!(n.v_basal, 0.0);
        assert_eq!(n.v_apical, 0.0);
    }

    #[test]
    fn astrocyte_calcium_rises_on_pre_spikes() {
        let mut n = AstrocyteLIFNeuron::new();
        let ca_before = n.ca;
        for _ in 0..100 {
            n.step_with_pre(0.0, true);
        }
        assert!(
            n.ca > ca_before,
            "Calcium must rise with presynaptic spikes"
        );
    }

    #[test]
    fn astrocyte_gliotransmitter_boosts_firing() {
        let mut n_no_glio = AstrocyteLIFNeuron::new();
        let mut n_glio = AstrocyteLIFNeuron::new();

        let mut spikes_no = 0;
        let mut spikes_yes = 0;
        for _ in 0..5000 {
            spikes_no += n_no_glio.step_with_pre(10.0, false);
            spikes_yes += n_glio.step_with_pre(10.0, true); // pre spikes → Ca → glio
        }
        assert!(
            spikes_yes >= spikes_no,
            "Gliotransmitter should boost firing: with={spikes_yes} >= without={spikes_no}"
        );
    }

    #[test]
    fn astrocyte_calcium_decays() {
        let mut n = AstrocyteLIFNeuron::new();
        // Build up calcium.
        for _ in 0..200 {
            n.step_with_pre(0.0, true);
        }
        let ca_peak = n.ca;
        // Let it decay.
        for _ in 0..5000 {
            n.step_with_pre(0.0, false);
        }
        assert!(
            n.ca < ca_peak * 0.5,
            "Calcium must decay: current={:.4} < peak={:.4}*0.5",
            n.ca,
            ca_peak
        );
    }
}
