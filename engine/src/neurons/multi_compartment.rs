// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

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
    pub gc: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_kdr: f64,
    pub g_ca: f64,
    pub g_kahp: f64,
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
            h: 0.9,
            n: 0.1,
            s: 0.0,
            c: 0.0,
            q: 0.0,
            gc: 2.1,
            p: 0.5,
            g_na: 30.0,
            g_kdr: 15.0,
            g_ca: 10.0,
            g_kahp: 0.8,
            g_l: 0.1,
            e_na: 60.0,
            e_k: -75.0,
            e_ca: 80.0,
            e_l: -60.0,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current_soma: f64, current_dend: f64) -> i32 {
        let v_prev = self.v_s;
        let am = safe_rate(0.32, 54.0, self.v_s, 4.0, 8.0);
        let bm = safe_rate(-0.28, 27.0, self.v_s, -5.0, 5.6);
        let m_inf = am / (am + bm);
        let ah = 0.128 * (-(self.v_s + 50.0) / 18.0).exp();
        let bh = 4.0 / (1.0 + (-(self.v_s + 27.0) / 5.0).exp());
        let an = safe_rate(0.032, 52.0, self.v_s, 5.0, 0.32);
        let bn = 0.5 * (-(self.v_s + 57.0) / 40.0).exp();
        let s_inf = 1.0 / (1.0 + (-(self.v_d + 20.0) / 9.0).exp());
        let i_na = self.g_na * m_inf.powi(2) * self.h * (self.v_s - self.e_na);
        let i_kdr = self.g_kdr * self.n.powi(2) * (self.v_s - self.e_k);
        let i_ls = self.g_l * (self.v_s - self.e_l);
        let i_ds = (self.gc / self.p) * (self.v_s - self.v_d);
        let i_ca = self.g_ca * self.s.powi(2) * (self.v_d - self.e_ca);
        let i_kahp = self.g_kahp * self.q * (self.v_d - self.e_k);
        let i_ld = self.g_l * (self.v_d - self.e_l);
        let i_sd = (self.gc / (1.0 - self.p)) * (self.v_d - self.v_s);
        self.v_s += (-i_na - i_kdr - i_ls - i_ds + current_soma / self.p) * self.dt;
        self.v_d += (-i_ca - i_kahp - i_ld - i_sd + current_dend / (1.0 - self.p)) * self.dt;
        self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
        self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
        self.s += ((s_inf - self.s) / 5.0) * self.dt;
        self.c = (self.c + (-0.13 * i_ca - 0.075 * self.c) * self.dt).max(0.0);
        let q_inf = (self.c / (self.c + 2.0)).min(1.0);
        self.q += ((q_inf - self.q) / 100.0) * self.dt;
        if self.v_s >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -60.0;
        self.v_d = -60.0;
        self.h = 0.9;
        self.n = 0.1;
        self.s = 0.0;
        self.c = 0.0;
        self.q = 0.0;
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
    pub fn step(&mut self, current_soma: f64, current_tuft: f64) -> i32 {
        let v_s_prev = self.v_s;
        for _ in 0..4 {
            let m_na_inf = 1.0 / (1.0 + (-(self.v_s + 38.0) / 7.0).exp());
            let h_na_inf = 1.0 / (1.0 + ((self.v_s + 65.0) / 6.0).exp());
            let n_k_inf = 1.0 / (1.0 + (-(self.v_s + 25.0) / 12.0).exp());
            let tau_h = 0.5 + 14.0 / (1.0 + ((self.v_s + 35.0) / 10.0).exp());
            let tau_n = 1.0 + 5.0 / (1.0 + ((self.v_s + 30.0) / 10.0).exp());
            self.h_na += (h_na_inf - self.h_na) / tau_h * self.dt;
            self.n_k += (n_k_inf - self.n_k) / tau_n * self.dt;
            let i_na = self.g_na * m_na_inf.powi(3) * self.h_na * (self.v_s - self.e_na);
            let i_k = self.g_k * self.n_k.powi(4) * (self.v_s - self.e_k);
            let i_l_s = self.g_l_s * (self.v_s - self.e_l);
            let i_st = self.g_st * (self.v_s - self.v_t) / self.p_s;
            let m_ca_inf = 1.0 / (1.0 + (-(self.v_t + 27.0) / 7.0).exp());
            let h_ca_inf = 1.0 / (1.0 + ((self.v_t + 52.0) / 5.0).exp());
            let m_ih_inf = 1.0 / (1.0 + ((self.v_t + 75.0) / 5.5).exp());
            self.m_ca += (m_ca_inf - self.m_ca) / 1.0 * self.dt;
            self.h_ca += (h_ca_inf - self.h_ca) / 20.0 * self.dt;
            self.m_ih += (m_ih_inf - self.m_ih) / 50.0 * self.dt;
            let i_ca_t = self.g_ca_t * self.m_ca.powi(2) * self.h_ca * (self.v_t - self.e_ca);
            let i_ih = self.g_ih * self.m_ih * (self.v_t - self.e_ih);
            let i_l_t = self.g_l_t * (self.v_t - self.e_l);
            let i_ts = self.g_st * (self.v_t - self.v_s) / self.p_t;
            let i_ta = self.g_ta * (self.v_t - self.v_a) / self.p_t;
            let m_ca_a_inf = 1.0 / (1.0 + (-(self.v_a + 30.0) / 5.0).exp());
            let kca_act = self.ca_a / (self.ca_a + 0.001);
            let i_ca_a = self.g_ca_a * m_ca_a_inf.powi(2) * (self.v_a - self.e_ca);
            let i_kca = self.g_kca * kca_act * (self.v_a - self.e_k);
            let i_l_a = self.g_l_a * (self.v_a - self.e_l);
            let i_at = self.g_ta * (self.v_a - self.v_t) / self.p_a;
            self.v_s += (-i_na - i_k - i_l_s - i_st + current_soma / self.p_s) / self.c_m * self.dt;
            self.v_t += (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / self.c_m * self.dt;
            self.v_a +=
                (-i_ca_a - i_kca - i_l_a - i_at + current_tuft / self.p_a) / self.c_m * self.dt;
            self.ca_a =
                (self.ca_a + (-self.f_ca * i_ca_a - self.ca_a / self.ca_decay) * self.dt).max(0.0);
        }
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
    pub m_a: f64,
    pub h_a: f64,
    pub m_kd: f64,
    pub m_h: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_cat: f64,
    pub g_cas: f64,
    pub g_a: f64,
    pub g_kd: f64,
    pub g_kca: f64,
    pub g_h: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl MarderSTGNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            m_na: 0.0,
            h_na: 0.9,
            m_cat: 0.0,
            h_cat: 0.9,
            m_cas: 0.0,
            m_a: 0.0,
            h_a: 0.9,
            m_kd: 0.0,
            m_h: 0.0,
            ca: 0.05,
            g_na: 400.0,
            g_cat: 2.5,
            g_cas: 6.0,
            g_a: 50.0,
            g_kd: 100.0,
            g_kca: 25.0,
            g_h: 0.02,
            g_l: 0.01,
            e_na: 50.0,
            e_k: -80.0,
            e_ca: 120.0,
            e_h: -20.0,
            e_l: -50.0,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let b = |v: f64, vh: f64, s: f64| 1.0 / (1.0 + (-(v - vh) / s).exp());
        self.m_na += (b(self.v, -25.5, 5.29) - self.m_na) / 1.32 * self.dt;
        self.h_na += (b(self.v, -48.9, -5.18) - self.h_na)
            / (0.67 * (1.0 + ((self.v + 62.9) / -10.0).exp()) + 1.5)
            * self.dt;
        self.m_cat += (b(self.v, -27.1, 7.2) - self.m_cat) / 21.7 * self.dt;
        self.h_cat += (b(self.v, -32.1, -5.5) - self.h_cat) / 105.0 * self.dt;
        self.m_cas += (b(self.v, -33.0, 8.1) - self.m_cas) / 14.0 * self.dt;
        self.m_a += (b(self.v, -27.2, 8.7) - self.m_a) / 11.6 * self.dt;
        self.h_a += (b(self.v, -56.9, -4.9) - self.h_a) / 38.6 * self.dt;
        self.m_kd += (b(self.v, -12.3, 11.8) - self.m_kd) / 7.2 * self.dt;
        self.m_h += (b(self.v, -70.0, -6.0) - self.m_h) / 272.0 * self.dt;
        let kca_act = (self.ca / (self.ca + 3.0)).min(1.0);
        let i_na = self.g_na * self.m_na.powi(3) * self.h_na * (self.v - self.e_na);
        let i_cat = self.g_cat * self.m_cat.powi(3) * self.h_cat * (self.v - self.e_ca);
        let i_cas = self.g_cas * self.m_cas.powi(3) * (self.v - self.e_ca);
        let i_a = self.g_a * self.m_a.powi(3) * self.h_a * (self.v - self.e_k);
        let i_kd = self.g_kd * self.m_kd.powi(4) * (self.v - self.e_k);
        let i_kca = self.g_kca * kca_act.powi(4) * (self.v - self.e_k);
        let i_h = self.g_h * self.m_h * (self.v - self.e_h);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_na - i_cat - i_cas - i_a - i_kd - i_kca - i_h - i_l + current) * self.dt;
        let i_ca_total = i_cat + i_cas;
        self.ca = (self.ca + (-0.0001 * i_ca_total - 0.01 * self.ca) * self.dt).max(0.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -60.0;
        self.m_na = 0.0;
        self.h_na = 0.9;
        self.m_cat = 0.0;
        self.h_cat = 0.9;
        self.m_cas = 0.0;
        self.m_a = 0.0;
        self.h_a = 0.9;
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
        Self {
            v: vec![-65.0; n_comp],
            n_comp,
            tau_m: 20.0,
            v_rest: -65.0,
            g_ratio: 0.5,
            v_threshold: -50.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let n = self.n_comp;
        let mut dv = vec![0.0; n];
        for i in 0..n {
            let leak = -(self.v[i] - self.v_rest) / self.tau_m;
            let left = if i > 0 {
                self.g_ratio * (self.v[i - 1] - self.v[i])
            } else {
                0.0
            };
            let right = if i + 1 < n {
                self.g_ratio * (self.v[i + 1] - self.v[i])
            } else {
                0.0
            };
            let inj = if i == n - 1 { current } else { 0.0 };
            dv[i] = (leak + left + right + inj) * self.dt;
        }
        for i in 0..n {
            self.v[i] += dv[i];
        }
        if self.v[0] >= self.v_threshold {
            self.v[0] = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v.fill(self.v_rest);
    }
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
            g_k: 100.0,
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
            let m_inf = 1.0 / (1.0 + (-(self.vs + 30.0) / 9.0).exp());
            let h_inf = 1.0 / (1.0 + ((self.vs + 45.0) / 7.0).exp());
            let n_inf = 1.0 / (1.0 + (-(self.vs + 30.0) / 10.0).exp());
            let m_ca_inf = 1.0 / (1.0 + (-(self.vd + 20.0) / 9.0).exp());
            let q_inf = (self.ca / (self.ca + 2.0)).min(1.0);
            self.h += (h_inf - self.h) / 1.0 * self.dt;
            self.n += (n_inf - self.n) / 3.0 * self.dt;
            self.q += (q_inf - self.q) / 100.0 * self.dt;
            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.vs - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.vs - self.e_k);
            let i_ls = self.g_l * (self.vs - self.e_l);
            let i_sd = (self.gc / self.p) * (self.vs - self.vd);
            let i_ca = self.g_ca * m_ca_inf.powi(2) * (self.vd - self.e_ca);
            let i_kca = self.g_kca * self.q * (self.vd - self.e_k);
            let i_ld = self.g_l * (self.vd - self.e_l);
            let i_ds = (self.gc / (1.0 - self.p)) * (self.vd - self.vs);
            self.vs += (-i_na - i_k - i_ls - i_sd + current / self.p) * self.dt;
            self.vd += (-i_ca - i_kca - i_ld - i_ds) * self.dt;
            self.ca = (self.ca + (-0.13 * i_ca - 0.075 * self.ca) * self.dt).max(0.0);
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
        let mut n = RallCableNeuron::new(5);
        let t: i32 = (0..500).map(|_| n.step(50.0)).sum();
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
}
