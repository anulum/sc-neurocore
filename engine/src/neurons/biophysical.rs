// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

//! Hodgkin-Huxley-type conductance-based neuron models.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

// ── Helper: safe alpha/beta kinetics avoiding division-by-zero ──

pub fn safe_rate(
    number_factor: f64,
    v_offset: f64,
    v: f64,
    denom_scale: f64,
    fallback: f64,
) -> f64 {
    let d = v + v_offset;
    if d.abs() < 1e-7 {
        fallback
    } else {
        number_factor * d / (1.0 - (-d / denom_scale).exp())
    }
}

/// Hodgkin-Huxley 1952 — 4-ODE ion channel model.
#[derive(Clone, Debug)]
pub struct HodgkinHuxleyNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub c_m: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HodgkinHuxleyNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
            dt: 0.01,
            v_threshold: 0.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let steps = (1.0 / self.dt) as usize;
        for _ in 0..steps {
            let am = safe_rate(0.1, 40.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 65.0) / 18.0).exp();
            let ah = 0.07 * (-(self.v + 65.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 35.0) / 10.0).exp());
            let an = safe_rate(0.01, 55.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 65.0) / 80.0).exp();
            self.m += (am * (1.0 - self.m) - bm * self.m) * self.dt;
            self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
            let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.32;
    }
}
impl Default for HodgkinHuxleyNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Traub-Miles — hippocampal pyramidal HH variant. Traub et al. 1991.
#[derive(Clone, Debug)]
pub struct TraubMilesNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
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

impl TraubMilesNeuron {
    pub fn new() -> Self {
        Self {
            v: -67.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            g_na: 100.0,
            g_k: 80.0,
            g_l: 0.1,
            e_na: 50.0,
            e_k: -100.0,
            e_l: -67.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..10 {
            let am = safe_rate(0.32, 54.0, self.v, 4.0, 8.0);
            let bm = safe_rate(-0.28, 27.0, self.v, -5.0, 5.6);
            let ah = 0.128 * (-(self.v + 50.0) / 18.0).exp();
            let bh = 4.0 / (1.0 + (-(self.v + 27.0) / 5.0).exp());
            let an = safe_rate(0.032, 52.0, self.v, 5.0, 0.32);
            let bn = 0.5 * (-(self.v + 57.0) / 40.0).exp();
            self.m += (am * (1.0 - self.m) - bm * self.m) * self.dt;
            self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
            let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_l + current) * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -67.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
    }
}
impl Default for TraubMilesNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Wang-Buzsaki — fast-spiking GABAergic interneuron. Wang & Buzsáki 1996.
#[derive(Clone, Debug)]
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
            v: -65.0,
            h: 0.8,
            n: 0.1,
            g_na: 35.0,
            g_k: 9.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..10 {
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();
            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;
            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.8;
        self.n = 0.1;
    }
}
impl Default for WangBuzsakiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Connor-Stevens — A-type K current for delay tuning. Connor et al. 1977.
#[derive(Clone, Debug)]
pub struct ConnorStevensNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub a: f64,
    pub b: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_a: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_a: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ConnorStevensNeuron {
    pub fn new() -> Self {
        Self {
            v: -68.0,
            m: 0.01,
            h: 0.99,
            n: 0.1,
            a: 0.5,
            b: 0.1,
            g_na: 120.0,
            g_k: 20.0,
            g_a: 47.7,
            g_l: 0.3,
            e_na: 55.0,
            e_k: -72.0,
            e_a: -75.0,
            e_l: -17.0,
            c_m: 1.0,
            dt: 0.01,
            v_threshold: 0.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..10 {
            let am = safe_rate(0.1, 29.7, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 54.7) / 18.0).exp();
            let ah = 0.07 * (-(self.v + 48.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 18.0) / 10.0).exp());
            let an = safe_rate(0.01, 45.7, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 55.7) / 80.0).exp();
            let a_inf = (0.0761 * ((self.v + 94.22) / 31.84).exp()
                / (1.0 + ((self.v + 1.17) / 28.93).exp()))
            .powf(1.0 / 3.0);
            let b_inf = 1.0 / (1.0 + ((self.v + 53.3) / 14.54).exp()).powf(4.0);
            let tau_a = 0.3632 + 1.158 / (1.0 + ((self.v + 55.96) / 20.12).exp());
            let tau_b = 1.24 + 2.678 / (1.0 + ((self.v + 50.0) / 16.027).exp());
            self.m += (am * (1.0 - self.m) - bm * self.m) * self.dt;
            self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
            self.a += (a_inf - self.a) / tau_a * self.dt;
            self.b += (b_inf - self.b) / tau_b * self.dt;
            let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_a = self.g_a * self.a.powi(3) * self.b * (self.v - self.e_a);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_a - i_l + current) / self.c_m * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -68.0;
        self.m = 0.01;
        self.h = 0.99;
        self.n = 0.1;
        self.a = 0.5;
        self.b = 0.1;
    }
}
impl Default for ConnorStevensNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Destexhe thalamocortical relay neuron with T-current. Destexhe et al. 1993.
#[derive(Clone, Debug)]
pub struct DestexheThalamicNeuron {
    pub v: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub m_t: f64,
    pub h_t: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_t: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl DestexheThalamicNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h_na: 0.6,
            n_k: 0.3,
            m_t: 0.0,
            h_t: 1.0,
            g_na: 100.0,
            g_k: 10.0,
            g_t: 2.0,
            g_l: 0.05,
            e_na: 50.0,
            e_k: -100.0,
            e_ca: 120.0,
            e_l: -70.0,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..5 {
            let m_na = 1.0 / (1.0 + (-(self.v + 37.0) / 7.0).exp());
            let h_na_inf = 1.0 / (1.0 + ((self.v + 41.0) / 4.0).exp());
            let n_inf = 1.0 / (1.0 + (-(self.v + 25.0) / 12.0).exp());
            let m_t_inf = 1.0 / (1.0 + (-(self.v + 57.0) / 6.2).exp());
            let h_t_inf = 1.0 / (1.0 + ((self.v + 81.0) / 4.0).exp());
            self.h_na += (h_na_inf - self.h_na) / 1.0 * self.dt;
            self.n_k += (n_inf - self.n_k) / 5.0 * self.dt;
            self.m_t += (m_t_inf - self.m_t) / 1.0 * self.dt;
            self.h_t += (h_t_inf - self.h_t) / 20.0 * self.dt;
            let i_na = self.g_na * m_na.powi(3) * self.h_na * (self.v - self.e_na);
            let i_k = self.g_k * self.n_k.powi(4) * (self.v - self.e_k);
            let i_t = self.g_t * self.m_t.powi(2) * self.h_t * (self.v - self.e_ca);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_t - i_l + current) * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h_na = 0.6;
        self.n_k = 0.3;
        self.m_t = 0.0;
        self.h_t = 1.0;
    }
}
impl Default for DestexheThalamicNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Huber-Braun — temperature-sensitive cold receptor. Braun et al. 1998.
#[derive(Clone, Debug)]
pub struct HuberBraunNeuron {
    pub v: f64,
    pub a_sd: f64,
    pub a_sr: f64,
    pub g_sd: f64,
    pub g_sr: f64,
    pub g_l: f64,
    pub e_sd: f64,
    pub e_sr: f64,
    pub e_l: f64,
    pub tau_sd: f64,
    pub tau_sr: f64,
    pub eta: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HuberBraunNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            a_sd: 0.0,
            a_sr: 0.0,
            g_sd: 1.5,
            g_sr: 0.4,
            g_l: 0.1,
            e_sd: 50.0,
            e_sr: -90.0,
            e_l: -60.0,
            tau_sd: 10.0,
            tau_sr: 20.0,
            eta: 0.012,
            dt: 0.1,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let sd_inf = 1.0 / (1.0 + (-(self.v + 25.0) / 5.0).exp());
        let sr_inf = 1.0 / (1.0 + ((self.v + 40.0) / 5.0).exp());
        self.a_sd += (sd_inf - self.a_sd) / self.tau_sd * self.dt;
        self.a_sr += (sr_inf - self.a_sr) / self.tau_sr * self.dt;
        let i_sd = self.g_sd * self.a_sd * (self.v - self.e_sd);
        let i_sr = self.g_sr * self.a_sr * (self.v - self.e_sr);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_sd - i_sr - i_l + current) * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.a_sd = 0.0;
        self.a_sr = 0.0;
    }
}
impl Default for HuberBraunNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Golomb fast-spiking interneuron with Kv3. Golomb et al. 2007.
#[derive(Clone, Debug)]
pub struct GolombFSNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_kv3: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl GolombFSNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.9,
            n: 0.1,
            p: 0.0,
            g_na: 112.5,
            g_k: 225.0,
            g_kv3: 150.0,
            g_l: 0.25,
            e_na: 50.0,
            e_k: -90.0,
            e_l: -70.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..10 {
            let m_inf = 1.0 / (1.0 + (-(self.v + 24.0) / 11.5).exp());
            let h_inf = 1.0 / (1.0 + ((self.v + 58.3) / 6.7).exp());
            let n_inf = 1.0 / (1.0 + (-(self.v + 12.4) / 6.8).exp());
            let p_inf = 1.0 / (1.0 + (-(self.v + 3.0) / 10.0).exp());
            self.h += (h_inf - self.h) / 0.5 * self.dt;
            self.n += (n_inf - self.n) / 2.0 * self.dt;
            self.p += (p_inf - self.p) / 1.0 * self.dt;
            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(2) * (self.v - self.e_k);
            let i_kv3 = self.g_kv3 * self.p * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_kv3 - i_l + current) * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h = 0.9;
        self.n = 0.1;
        self.p = 0.0;
    }
}
impl Default for GolombFSNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Pospischil — minimal HH for 5 cortical cell types. Pospischil et al. 2008.
#[derive(Clone, Debug)]
pub struct PospischilNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_m: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub vt: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PospischilNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            p: 0.0,
            g_na: 50.0,
            g_k: 5.0,
            g_m: 0.004,
            g_l: 0.01,
            e_na: 50.0,
            e_k: -90.0,
            e_l: -70.0,
            vt: -56.2,
            dt: 0.025,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..4 {
            let dv = self.v - self.vt;
            let am = safe_rate(-0.32, 0.0, dv - 13.0, -4.0, 8.0);
            let bm = safe_rate(0.28, 0.0, dv - 40.0, 5.0, 5.6);
            let ah = 0.128 * (-(dv - 17.0) / 18.0).exp();
            let bh = 4.0 / (1.0 + (-(dv - 40.0) / 5.0).exp());
            let an = safe_rate(-0.032, 0.0, dv - 15.0, -5.0, 0.32);
            let bn = 0.5 * (-(dv - 10.0) / 40.0).exp();
            let p_inf = 1.0 / (1.0 + (-(self.v + 35.0) / 10.0).exp());
            self.m += (am * (1.0 - self.m) - bm * self.m) * self.dt;
            self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
            self.p += (p_inf - self.p) / 100.0 * self.dt;
            let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_m = self.g_m * self.p * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_m - i_l + current) * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -70.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
        self.p = 0.0;
    }
}
impl Default for PospischilNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Mainen-Sejnowski — two-compartment (soma + axon). Mainen & Sejnowski 1996.
#[derive(Clone, Debug)]
pub struct MainenSejnowskiNeuron {
    pub vs: f64,
    pub va: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub kappa: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_s: f64,
    pub c_a: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl MainenSejnowskiNeuron {
    pub fn new() -> Self {
        Self {
            vs: -65.0,
            va: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            kappa: 10.0,
            g_na: 3000.0,
            g_k: 1500.0,
            g_l: 1.0,
            e_na: 50.0,
            e_k: -90.0,
            e_l: -70.0,
            c_s: 1.0,
            c_a: 0.1,
            dt: 0.005,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.vs;
        for _ in 0..20 {
            let am = safe_rate(0.1, 40.0, self.va, 10.0, 1.0);
            let bm = 4.0 * (-(self.va + 65.0) / 18.0).exp();
            let ah = 0.07 * (-(self.va + 65.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.va + 35.0) / 10.0).exp());
            let an = safe_rate(0.01, 55.0, self.va, 10.0, 0.1);
            let bn = 0.125 * (-(self.va + 65.0) / 80.0).exp();
            self.m += (am * (1.0 - self.m) - bm * self.m) * self.dt;
            self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
            let i_na = self.g_na * self.m.powi(3) * self.h * (self.va - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.va - self.e_k);
            let i_l_s = self.g_l * (self.vs - self.e_l);
            let i_c = self.kappa * (self.va - self.vs);
            self.vs += (-i_l_s + i_c + current) / self.c_s * self.dt;
            self.va += (-i_na - i_k - self.kappa * (self.va - self.vs)) / self.c_a * self.dt;
        }
        if self.vs >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.vs = -65.0;
        self.va = -65.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
    }
}
impl Default for MainenSejnowskiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// De Schutter-Bower Purkinje cell — Ca-dependent K. De Schutter & Bower 1994.
#[derive(Clone, Debug)]
pub struct DeSchutterPurkinjeNeuron {
    pub v: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub m_cap: f64,
    pub h_cap: f64,
    pub q_kca: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_cap: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub ca_decay: f64,
    pub f_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl DeSchutterPurkinjeNeuron {
    pub fn new() -> Self {
        Self {
            v: -68.0,
            h_na: 0.8,
            n_k: 0.1,
            m_cap: 0.0,
            h_cap: 0.9,
            q_kca: 0.0,
            ca: 0.0001,
            g_na: 125.0,
            g_k: 10.0,
            g_cap: 45.0,
            g_kca: 35.0,
            g_l: 0.5,
            e_na: 45.0,
            e_k: -85.0,
            e_ca: 135.0,
            e_l: -68.0,
            ca_decay: 0.02,
            f_ca: 0.00024,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..5 {
            let m_na = 1.0 / (1.0 + (-(self.v + 30.0) / 9.0).exp());
            let h_na_inf = 1.0 / (1.0 + ((self.v + 45.0) / 7.0).exp());
            let n_inf = 1.0 / (1.0 + (-(self.v + 25.0) / 12.0).exp());
            let m_cap_inf = 1.0 / (1.0 + (-(self.v + 19.0) / 10.0).exp());
            let h_cap_inf = 1.0 / (1.0 + ((self.v + 39.0) / 7.0).exp());
            let q_inf = self.ca / (self.ca + 0.001);
            self.h_na += (h_na_inf - self.h_na) / 1.0 * self.dt;
            self.n_k += (n_inf - self.n_k) / 3.0 * self.dt;
            self.m_cap += (m_cap_inf - self.m_cap) / 1.0 * self.dt;
            self.h_cap += (h_cap_inf - self.h_cap) / 15.0 * self.dt;
            self.q_kca += (q_inf - self.q_kca) / 5.0 * self.dt;
            let i_na = self.g_na * m_na.powi(3) * self.h_na * (self.v - self.e_na);
            let i_k = self.g_k * self.n_k.powi(4) * (self.v - self.e_k);
            let i_cap = self.g_cap * self.m_cap.powi(2) * self.h_cap * (self.v - self.e_ca);
            let i_kca = self.g_kca * self.q_kca * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_cap - i_kca - i_l + current) * self.dt;
            self.ca = (self.ca + (-self.f_ca * i_cap - self.ca_decay * self.ca) * self.dt).max(0.0);
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -68.0;
        self.h_na = 0.8;
        self.n_k = 0.1;
        self.m_cap = 0.0;
        self.h_cap = 0.9;
        self.q_kca = 0.0;
        self.ca = 0.0001;
    }
}
impl Default for DeSchutterPurkinjeNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Plant R15 — Aplysia parabolic burster. Plant & Kim 1976.
#[derive(Clone, Debug)]
pub struct PlantR15Neuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_ca: f64,
    pub g_l: f64,
    pub g_kca: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub k_ca: f64,
    pub tau_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PlantR15Neuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            m: 0.05,
            h: 0.6,
            n: 0.3,
            ca: 0.1,
            g_na: 4.0,
            g_k: 0.3,
            g_ca: 0.004,
            g_l: 0.003,
            g_kca: 0.03,
            e_na: 30.0,
            e_k: -75.0,
            e_ca: 140.0,
            e_l: -40.0,
            c_m: 1.0,
            k_ca: 0.0085,
            tau_ca: 500.0,
            dt: 0.05,
            v_threshold: -10.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        for _ in 0..5 {
            let am = safe_rate(0.1, 40.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 65.0) / 18.0).exp();
            let ah = 0.07 * (-(self.v + 65.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 35.0) / 10.0).exp());
            let an = safe_rate(0.01, 55.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 65.0) / 80.0).exp();
            self.m += (am * (1.0 - self.m) - bm * self.m) * self.dt;
            self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
            let m_ca = 1.0 / (1.0 + (-(self.v + 25.0) / 5.0).exp());
            let kca_act = (self.ca / (self.ca + 0.5)).min(1.0);
            let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_ca = self.g_ca * m_ca * (self.v - self.e_ca);
            let i_kca = self.g_kca * kca_act * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_ca - i_kca - i_l + current) / self.c_m * self.dt;
            self.ca = (self.ca + (-self.k_ca * i_ca - self.ca / self.tau_ca) * self.dt).max(0.0);
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.3;
        self.ca = 0.1;
    }
}
impl Default for PlantR15Neuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Prescott 2008 — Type I/II/III excitability tuning via M-current.
#[derive(Clone, Debug)]
pub struct PrescottNeuron {
    pub v: f64,
    pub w: f64,
    pub g_fast: f64,
    pub g_slow: f64,
    pub g_l: f64,
    pub e_fast: f64,
    pub e_slow: f64,
    pub e_l: f64,
    pub beta_w: f64,
    pub gamma_w: f64,
    pub tau_w: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PrescottNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            g_fast: 20.0,
            g_slow: 20.0,
            g_l: 2.0,
            e_fast: 50.0,
            e_slow: -100.0,
            e_l: -70.0,
            beta_w: -21.0,
            gamma_w: 15.0,
            tau_w: 100.0,
            phi: 0.15,
            dt: 0.1,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 20.0) / 15.0).exp());
        let w_inf = 1.0 / (1.0 + (-(self.v - self.beta_w) / self.gamma_w).exp());
        let i_fast = self.g_fast * m_inf * (self.v - self.e_fast);
        let i_slow = self.g_slow * self.w * (self.v - self.e_slow);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_fast - i_slow - i_l + current) * self.dt;
        self.w += self.phi * (w_inf - self.w) / self.tau_w * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.w = 0.0;
    }
}
impl Default for PrescottNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Mihalas-Niebur 2009 — generalized IF capturing 20 spike patterns.
#[derive(Clone, Debug)]
pub struct MihalasNieburNeuron {
    pub v: f64,
    pub theta: f64,
    pub i1: f64,
    pub i2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta_reset: f64,
    pub theta_inf: f64,
    pub tau_v: f64,
    pub tau_theta: f64,
    pub tau_1: f64,
    pub tau_2: f64,
    pub a: f64,
    pub b: f64,
    pub r1: f64,
    pub r2: f64,
    pub dt: f64,
}

impl MihalasNieburNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            theta: 1.0,
            i1: 0.0,
            i2: 0.0,
            v_rest: 0.0,
            v_reset: 0.0,
            theta_reset: 1.0,
            theta_inf: 1.0,
            tau_v: 10.0,
            tau_theta: 100.0,
            tau_1: 10.0,
            tau_2: 200.0,
            a: 0.0,
            b: 0.0,
            r1: 0.0,
            r2: 0.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + self.i1 + self.i2 + current) / self.tau_v * self.dt;
        self.theta += self.a * (self.v - self.v_rest)
            + (self.theta_inf - self.theta) / self.tau_theta * self.dt;
        self.i1 += -self.i1 / self.tau_1 * self.dt;
        self.i2 += -self.i2 / self.tau_2 * self.dt;
        if self.v >= self.theta {
            self.v = self.v_reset + self.b * (self.v - self.v_rest);
            self.theta = self.theta_reset.max(self.theta);
            self.i1 += self.r1;
            self.i2 += self.r2;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_reset;
        self.i1 = 0.0;
        self.i2 = 0.0;
    }
}
impl Default for MihalasNieburNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Allen GLIF5 — LIF + threshold adaptation + after-spike currents.
#[derive(Clone, Debug)]
pub struct GLIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub theta_inf: f64,
    pub i_asc1: f64,
    pub i_asc2: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub tau_asc1: f64,
    pub tau_asc2: f64,
    pub a_theta: f64,
    pub delta_theta: f64,
    pub r_asc1: f64,
    pub r_asc2: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl GLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            theta: -50.0,
            theta_inf: -50.0,
            i_asc1: 0.0,
            i_asc2: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            tau_m: 10.0,
            tau_theta: 100.0,
            tau_asc1: 10.0,
            tau_asc2: 200.0,
            a_theta: 0.01,
            delta_theta: 2.0,
            r_asc1: 1.0,
            r_asc2: 0.5,
            resistance: 1.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + self.resistance * current + self.i_asc1 + self.i_asc2)
            / self.tau_m
            * self.dt;
        self.theta += self.a_theta * (self.v - self.v_rest)
            + (self.theta_inf - self.theta) / self.tau_theta * self.dt;
        self.i_asc1 *= (-self.dt / self.tau_asc1).exp();
        self.i_asc2 *= (-self.dt / self.tau_asc2).exp();
        if self.v >= self.theta {
            self.v = self.v_reset;
            self.theta += self.delta_theta;
            self.i_asc1 += self.r_asc1;
            self.i_asc2 += self.r_asc2;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_inf;
        self.i_asc1 = 0.0;
        self.i_asc2 = 0.0;
    }
}
impl Default for GLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// GIF population — escape-rate generalized IF. Mensi et al. 2012.
#[derive(Clone, Debug)]
pub struct GIFPopulationNeuron {
    pub v: f64,
    pub theta: f64,
    pub eta: f64,
    pub tau_m: f64,
    pub tau_eta: f64,
    pub delta_v: f64,
    pub lambda_0: f64,
    pub eta_increment: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub dt: f64,
    rng: Xoshiro256PlusPlus,
}

impl GIFPopulationNeuron {
    pub fn new(seed: u64) -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            eta: 0.0,
            tau_m: 20.0,
            tau_eta: 100.0,
            delta_v: 2.0,
            lambda_0: 0.001,
            eta_increment: 5.0,
            v_rest: -65.0,
            v_reset: -65.0,
            dt: 0.5,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + current) / self.tau_m * self.dt;
        self.eta *= (-self.dt / self.tau_eta).exp();
        let hazard = self.lambda_0 * ((self.v - self.theta + self.eta) / self.delta_v).exp();
        if self.rng.random::<f64>() < hazard * self.dt {
            self.v = self.v_reset;
            self.eta += self.eta_increment;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.eta = 0.0;
    }
}

/// Av-Ron cardiac ganglion — Type III burster. Av-Ron et al. 1991.
#[derive(Clone, Debug)]
pub struct AvRonCardiacNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub s: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_s: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_s: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl AvRonCardiacNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            h: 0.6,
            n: 0.3,
            s: 0.5,
            g_na: 80.0,
            g_k: 40.0,
            g_s: 20.0,
            g_l: 0.1,
            e_na: 40.0,
            e_k: -80.0,
            e_s: -25.0,
            e_l: -60.0,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 35.0) / 7.8).exp());
        let h_inf = 1.0 / (1.0 + ((self.v + 55.0) / 7.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 28.0) / 15.0).exp());
        let s_inf = 1.0 / (1.0 + (-(self.v + 27.0) / 5.0).exp());
        self.h += (h_inf - self.h) / 1.5 * self.dt;
        self.n += (n_inf - self.n) / 4.0 * self.dt;
        self.s += (s_inf - self.s) / 50.0 * self.dt;
        let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_s = self.g_s * self.s * (self.v - self.e_s);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_na - i_k - i_s - i_l + current) * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -60.0;
        self.h = 0.6;
        self.n = 0.3;
        self.s = 0.5;
    }
}
impl Default for AvRonCardiacNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Durstewitz PFC neuron with D1 dopamine modulation. Durstewitz et al. 2000.
#[derive(Clone, Debug)]
pub struct DurstewitzDopamineNeuron {
    pub v: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_nmda: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_nmda: f64,
    pub e_l: f64,
    pub mg: f64,
    pub d1_level: f64,
    pub g_nmda_scale: f64,
    pub g_k_scale: f64,
    pub v_shift_na: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl DurstewitzDopamineNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h_na: 0.7,
            n_k: 0.2,
            g_na: 45.0,
            g_k: 18.0,
            g_nmda: 0.5,
            g_l: 0.02,
            e_na: 55.0,
            e_k: -80.0,
            e_nmda: 0.0,
            e_l: -65.0,
            mg: 1.0,
            d1_level: 0.0,
            g_nmda_scale: 2.5,
            g_k_scale: 1.5,
            v_shift_na: -5.0,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let shift = self.d1_level * self.v_shift_na;
        let m_inf = 1.0 / (1.0 + (-(self.v + 30.0 + shift) / 9.5).exp());
        let h_inf = 1.0 / (1.0 + ((self.v + 53.0) / 7.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 30.0) / 10.0).exp());
        self.h_na += (h_inf - self.h_na) / 1.0 * self.dt;
        self.n_k += (n_inf - self.n_k) / 4.0 * self.dt;
        let g_eff_nmda = self.g_nmda * (1.0 + self.d1_level * (self.g_nmda_scale - 1.0));
        let g_eff_k = self.g_k * (1.0 + self.d1_level * (self.g_k_scale - 1.0));
        let mg_block = 1.0 / (1.0 + self.mg * (-0.062 * self.v).exp() / 3.57);
        let i_na = self.g_na * m_inf.powi(3) * self.h_na * (self.v - self.e_na);
        let i_k = g_eff_k * self.n_k.powi(4) * (self.v - self.e_k);
        let i_nmda = g_eff_nmda * mg_block * (self.v - self.e_nmda);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_na - i_k - i_nmda - i_l + current) * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h_na = 0.7;
        self.n_k = 0.2;
    }
}
impl Default for DurstewitzDopamineNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Hill-Tononi 2005 — thalamocortical sleep/wake with Na-dependent K.
#[derive(Clone, Debug)]
pub struct HillTononiNeuron {
    pub v: f64,
    pub h_na: f64,
    pub n_k: f64,
    pub m_h: f64,
    pub h_t: f64,
    pub na_i: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_h: f64,
    pub g_t: f64,
    pub g_kna: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_h: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub na_pump_max: f64,
    pub na_eq: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HillTononiNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h_na: 0.6,
            n_k: 0.3,
            m_h: 0.0,
            h_t: 0.9,
            na_i: 5.0,
            g_na: 50.0,
            g_k: 5.0,
            g_h: 1.0,
            g_t: 3.0,
            g_kna: 1.33,
            g_l: 0.02,
            e_na: 50.0,
            e_k: -90.0,
            e_h: -43.0,
            e_ca: 120.0,
            e_l: -70.0,
            na_pump_max: 20.0,
            na_eq: 9.5,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_na_inf = 1.0 / (1.0 + (-(self.v + 38.0) / 10.0).exp());
        let h_na_inf = 1.0 / (1.0 + ((self.v + 43.0) / 6.0).exp());
        let n_k_inf = 1.0 / (1.0 + (-(self.v + 27.0) / 11.5).exp());
        let m_h_inf = 1.0 / (1.0 + ((self.v + 75.0) / 5.5).exp());
        let m_t_inf = 1.0 / (1.0 + (-(self.v + 59.0) / 6.2).exp());
        let h_t_inf = 1.0 / (1.0 + ((self.v + 83.0) / 4.0).exp());
        let w_kna = 0.37 / (1.0 + (38.7 / self.na_i.max(0.01)).powf(3.5));
        let tau_h_na = (1.0 + 10.0 / (1.0 + ((self.v + 40.0) / 10.0).exp())).max(0.1);
        let tau_n_k = (5.0 + 47.0 * (-(((self.v + 50.0) / 25.0).powi(2))))
            .max(0.1)
            .exp();
        let tau_m_h = (20.0
            + 1000.0 / (((self.v + 71.5) / 14.2).exp() + (-(self.v + 89.0) / 11.6).exp()))
        .max(1.0);
        let tau_h_t = if self.v < -81.0 {
            (30.8 + 211.4 * ((self.v + 115.2) / 5.0).exp() / (1.0 + ((self.v + 86.0) / 3.2).exp()))
                .max(0.1)
        } else {
            10.0
        };
        self.h_na += (h_na_inf - self.h_na) / tau_h_na * self.dt;
        self.n_k += (n_k_inf - self.n_k) / tau_n_k * self.dt;
        self.m_h += (m_h_inf - self.m_h) / tau_m_h * self.dt;
        self.h_t += (h_t_inf - self.h_t) / tau_h_t * self.dt;
        let i_na = self.g_na * m_na_inf.powi(3) * self.h_na * (self.v - self.e_na);
        let i_k = self.g_k * self.n_k.powi(4) * (self.v - self.e_k);
        let i_h = self.g_h * self.m_h * (self.v - self.e_h);
        let i_t = self.g_t * m_t_inf.powi(2) * self.h_t * (self.v - self.e_ca);
        let i_kna = self.g_kna * w_kna * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_na - i_k - i_h - i_t - i_kna - i_l + current) * self.dt;
        self.na_i = (self.na_i
            + (-0.001 * i_na - self.na_pump_max * (self.na_i / (self.na_i + self.na_eq)))
                * self.dt)
            .max(0.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.h_na = 0.6;
        self.n_k = 0.3;
        self.m_h = 0.0;
        self.h_t = 0.9;
        self.na_i = 5.0;
    }
}
impl Default for HillTononiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Bertram phantom burster — dual slow K for phantom bursting. Bertram et al. 2000.
#[derive(Clone, Debug)]
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
            v_s1: -35.0,
            s_s1: 10.0,
            v_s2: -35.0,
            s_s2: 10.0,
            tau_s1: 20000.0,
            tau_s2: 100000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v - self.v_m) / self.s_m).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v - self.v_n) / self.s_n).exp());
        let s1_inf = 1.0 / (1.0 + (-(self.v - self.v_s1) / self.s_s1).exp());
        let s2_inf = 1.0 / (1.0 + (-(self.v - self.v_s2) / self.s_s2).exp());
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * n_inf * (self.v - self.e_k);
        let i_s1 = self.g_s1 * self.s1 * (self.v - self.e_k);
        let i_s2 = self.g_s2 * self.s2 * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / self.c_m * self.dt;
        self.s1 += (s1_inf - self.s1) / self.tau_s1 * self.dt;
        self.s2 += (s2_inf - self.s2) / self.tau_s2 * self.dt;
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
impl Default for BertramPhantomBurster {
    fn default() -> Self {
        Self::new()
    }
}

/// Yamada 1989 — subcritical Hopf burster.
#[derive(Clone, Debug)]
pub struct YamadaNeuron {
    pub v: f64,
    pub n: f64,
    pub q: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_q: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_q: f64,
    pub e_l: f64,
    pub tau_q: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl YamadaNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            n: 0.1,
            q: 0.0,
            g_na: 20.0,
            g_k: 10.0,
            g_q: 5.0,
            g_l: 0.5,
            e_na: 60.0,
            e_k: -80.0,
            e_q: -80.0,
            e_l: -60.0,
            tau_q: 300.0,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 30.0) / 9.5).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 30.0) / 10.0).exp());
        let q_inf = 1.0 / (1.0 + (-(self.v + 50.0) / 10.0).exp());
        let tau_n = 1.0 + 7.5 / (1.0 + ((self.v + 40.0) / 12.0).exp());
        let i_na = self.g_na * m_inf.powi(3) * (1.0 - self.n) * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_q = self.g_q * self.q * (self.v - self.e_q);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_na - i_k - i_q - i_l + current) * self.dt;
        self.n += (n_inf - self.n) / tau_n * self.dt;
        self.q += (q_inf - self.q) / self.tau_q * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -60.0;
        self.n = 0.1;
        self.q = 0.0;
    }
}
impl Default for YamadaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hh_fires() {
        let mut n = HodgkinHuxleyNeuron::new();
        let t: i32 = (0..100).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn traub_fires() {
        let mut n = TraubMilesNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn wb_fires() {
        let mut n = WangBuzsakiNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn cs_fires() {
        let mut n = ConnorStevensNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn destexhe_fires() {
        let mut n = DestexheThalamicNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn hb_fires() {
        let mut n = HuberBraunNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn golomb_fires() {
        let mut n = GolombFSNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(200.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn pospischil_fires() {
        let mut n = PospischilNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn mainen_fires() {
        let mut n = MainenSejnowskiNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(500.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn purkinje_fires() {
        let mut n = DeSchutterPurkinjeNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(50.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn plant_r15_fires() {
        let mut n = PlantR15Neuron::new();
        let t: i32 = (0..500).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn prescott_fires() {
        let mut n = PrescottNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn mn_fires() {
        let mut n = MihalasNieburNeuron::new();
        let t: i32 = (0..100).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn glif_fires() {
        let mut n = GLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(30.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn gif_pop_fires() {
        let mut n = GIFPopulationNeuron::new(42);
        let t: i32 = (0..1000).map(|_| n.step(30.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn avron_fires() {
        let mut n = AvRonCardiacNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn durstewitz_fires() {
        let mut n = DurstewitzDopamineNeuron::new();
        let t: i32 = (0..1000).map(|_| n.step(3.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn hill_tononi_fires() {
        let mut n = HillTononiNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn bertram_fires() {
        let mut n = BertramPhantomBurster::new();
        let t: i32 = (0..10000).map(|_| n.step(200.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn yamada_fires() {
        let mut n = YamadaNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }
}
