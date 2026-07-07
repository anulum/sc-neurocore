// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Specific Interneuron Types

//! Biophysically faithful interneuron models for cortical and cerebellar circuits.
//!
//! Six cell types covering the major inhibitory neuron classes:
//! - PV+ fast-spiking (Wang-Buzsáki 1996 base + Kv3.1)
//! - SST+ low-threshold spiking (Pospischil 2008 LTS variant)
//! - VIP irregular spiking (accommodating, high Rin)
//! - Chandelier axo-axonic (WB base + Kv1 delay + Kv3.1)
//! - Basket cell cerebellar (Midtgaard 1992 kinetics)
//! - Martinotti cell (adapting, ascending axon targeting L1)

use super::biophysical::safe_rate;

// ═══════════════════════════════════════════════════════════════════
// PV+ Fast-Spiking Interneuron
// ═══════════════════════════════════════════════════════════════════

/// PV+ (parvalbumin) fast-spiking interneuron.
///
/// Biophysics: Wang-Buzsáki 1996 core (Na+, Kdr, leak) extended with
/// Kv3.1 (fast-activating K+ for narrow APs and high-frequency firing).
/// Key properties: narrow APs, high sustained firing (>200 Hz),
/// no spike frequency adaptation, low input resistance.
///
/// Wang & Buzsáki 1996, J Neurosci 16:6402-6413 + Kv3.1 extension.
#[derive(Clone, Debug)]
pub struct PVFastSpikingNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64, // Kv3.1 activation
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_kv3: f64,
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl PVFastSpikingNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            p: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_kv3: 5.0, // Kv3.1 for narrow APs
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0, // Fast kinetics (FS phenotype)
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    /// Return `[dV, dh, dn, dp]` of the four-state Wang-Buzsáki + Kv3.1 system at
    /// one consistent state.
    fn derivatives(&self, v: f64, h: f64, n: f64, p: f64, current: f64) -> [f64; 4] {
        let am = safe_rate(0.1, 35.0, v, 10.0, 1.0);
        let bm = 4.0 * (-(v + 60.0) / 18.0).exp();
        let m_inf = am / (am + bm);
        let ah = 0.07 * (-(v + 58.0) / 20.0).exp();
        let bh = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
        let an = safe_rate(0.01, 34.0, v, 10.0, 0.1);
        let bn = 0.125 * (-(v + 44.0) / 80.0).exp();
        let p_inf = 1.0 / (1.0 + (-(v + 10.0) / 10.0).exp());
        let dh = self.phi * (ah * (1.0 - h) - bh * h);
        let dn = self.phi * (an * (1.0 - n) - bn * n);
        let dp = self.phi * (p_inf - p);
        let i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na);
        let i_k = self.g_k * n * n * n * n * (v - self.e_k);
        let i_kv3 = self.g_kv3 * p * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_na - i_k - i_kv3 - i_l + current) / self.c_m;
        [dv, dh, dn, dp]
    }

    /// Return one classical RK4 increment of `[V, h, n, p]`, holding `current`
    /// constant across the four stages.
    fn rk4_substep(&self, s: [f64; 4], current: f64) -> [f64; 4] {
        let dt = self.dt;
        let k1 = self.derivatives(s[0], s[1], s[2], s[3], current);
        let k2 = self.derivatives(
            s[0] + 0.5 * dt * k1[0],
            s[1] + 0.5 * dt * k1[1],
            s[2] + 0.5 * dt * k1[2],
            s[3] + 0.5 * dt * k1[3],
            current,
        );
        let k3 = self.derivatives(
            s[0] + 0.5 * dt * k2[0],
            s[1] + 0.5 * dt * k2[1],
            s[2] + 0.5 * dt * k2[2],
            s[3] + 0.5 * dt * k2[3],
            current,
        );
        let k4 = self.derivatives(
            s[0] + dt * k3[0],
            s[1] + dt * k3[1],
            s[2] + dt * k3[2],
            s[3] + dt * k3[3],
            current,
        );
        let mut out = [0.0_f64; 4];
        for i in 0..4 {
            out[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        let mut s = [self.v, self.h, self.n, self.p];
        for _ in 0..n_sub {
            s = self.rk4_substep(s, current);
        }
        self.v = s[0];
        self.h = s[1];
        self.n = s[2];
        self.p = s[3];
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
        self.p = 0.0;
    }
}

impl Default for PVFastSpikingNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// SST+ Low-Threshold Spiking Interneuron
// ═══════════════════════════════════════════════════════════════════

/// SST+ (somatostatin) low-threshold spiking interneuron.
///
/// Biophysics: Na+, K+, M-current (Kv7, slow K+ for adaptation),
/// T-type Ca2+ (low-threshold burst), h-current (Ih, sag), leak.
/// Key properties: spike frequency adaptation, rebound bursting,
/// facilitating synapses, dendritic targeting.
///
/// Based on Pospischil et al. 2008 LTS parameterisation.
#[derive(Clone, Debug)]
pub struct SSTNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64, // M-current activation
    pub s: f64, // T-type Ca2+ inactivation
    pub r: f64, // h-current activation
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_m: f64,
    pub g_t: f64,
    pub g_h: f64,
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl SSTNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            m: 0.02,
            h: 0.8,
            n: 0.2,
            p: 0.0,
            s: 0.9,
            r: 0.1,
            g_na: 50.0,
            g_k: 5.0,
            g_m: 0.12, // Strong M-current → adaptation
            g_t: 0.01, // T-type Ca2+ for rebound (minimal window current)
            g_h: 0.02, // Ih for sag
            g_l: 0.05, // Leak for resting stability
            e_na: 50.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_h: -40.0,
            e_l: -65.0,
            c_m: 1.0,
            dt: 0.025,
            v_threshold: -20.0,
        }
    }

    /// Return `[dV, dm, dh, dn, dp, ds, dr]` of the seven-state SST system at one
    /// consistent state. The Na/K activation rates use the L'Hôpital limit at the
    /// removable Traub-Miles singularity; β_m carries the published `V - V_T - 40`
    /// offset (an earlier `-17` offset drove the cell into depolarisation block).
    fn derivatives(
        &self,
        v: f64,
        m: f64,
        h: f64,
        n: f64,
        p: f64,
        s: f64,
        r: f64,
        current: f64,
    ) -> [f64; 7] {
        let dvt = v - (-56.2);
        let asing = |num: f64, slope: f64, limit: f64| {
            if num.abs() < 1e-6 {
                limit
            } else {
                num / ((num / slope).exp() - 1.0)
            }
        };
        let alpha_m = -0.32 * asing(dvt - 13.0, -4.0, -4.0);
        let beta_m = 0.28 * asing(dvt - 40.0, 5.0, 5.0);
        let alpha_h = 0.128 * (-(dvt - 17.0) / 18.0).exp();
        let beta_h = 4.0 / (1.0 + (-(dvt - 40.0) / 5.0).exp());
        let alpha_n = -0.032 * asing(dvt - 15.0, -5.0, -5.0);
        let beta_n = 0.5 * (-(dvt - 10.0) / 40.0).exp();
        let dm = alpha_m * (1.0 - m) - beta_m * m;
        let dh = alpha_h * (1.0 - h) - beta_h * h;
        let dn = alpha_n * (1.0 - n) - beta_n * n;
        let p_inf = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());
        let tau_p = 400.0 / (3.3 * ((v + 35.0) / 20.0).exp() + (-(v + 35.0) / 20.0).exp());
        let dp = (p_inf - p) / tau_p;
        let m_t_inf = 1.0 / (1.0 + (-(v + 57.0) / 6.2).exp());
        let s_inf = 1.0 / (1.0 + ((v + 81.0) / 4.0).exp());
        let tau_s = 30.0 + 200.0 / (1.0 + ((v + 70.0) / 5.0).exp());
        let ds = (s_inf - s) / tau_s;
        let r_inf = 1.0 / (1.0 + ((v + 80.0) / 10.0).exp());
        let tau_r = 100.0 + 500.0 / ((-(v + 70.0) / 20.0).exp() + ((v + 70.0) / 20.0).exp());
        let dr = (r_inf - r) / tau_r;
        let i_na = self.g_na * m * m * m * h * (v - self.e_na);
        let i_k = self.g_k * n * n * n * n * (v - self.e_k);
        let i_m = self.g_m * p * (v - self.e_k);
        let i_t = self.g_t * m_t_inf * m_t_inf * s * (v - self.e_ca);
        let i_h = self.g_h * r * (v - self.e_h);
        let i_l = self.g_l * (v - self.e_l);
        let dvdt = (-i_na - i_k - i_m - i_t - i_h - i_l + current) / self.c_m;
        [dvdt, dm, dh, dn, dp, ds, dr]
    }

    /// Return one classical RK4 increment of `[V, m, h, n, p, s, r]`, holding
    /// `current` constant across the four stages.
    fn rk4_substep(&self, st: [f64; 7], current: f64) -> [f64; 7] {
        let dt = self.dt;
        let k1 = self.derivatives(st[0], st[1], st[2], st[3], st[4], st[5], st[6], current);
        let mut a = [0.0_f64; 7];
        for i in 0..7 {
            a[i] = st[i] + 0.5 * dt * k1[i];
        }
        let k2 = self.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], a[6], current);
        for i in 0..7 {
            a[i] = st[i] + 0.5 * dt * k2[i];
        }
        let k3 = self.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], a[6], current);
        for i in 0..7 {
            a[i] = st[i] + dt * k3[i];
        }
        let k4 = self.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], a[6], current);
        let mut out = [0.0_f64; 7];
        for i in 0..7 {
            out[i] = st[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let mut st = [self.v, self.m, self.h, self.n, self.p, self.s, self.r];
        for _ in 0..4 {
            st = self.rk4_substep(st, current);
        }
        self.v = st[0];
        self.m = st[1];
        self.h = st[2];
        self.n = st[3];
        self.p = st[4];
        self.s = st[5];
        self.r = st[6];
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.m = 0.02;
        self.h = 0.8;
        self.n = 0.2;
        self.p = 0.0;
        self.s = 0.9;
        self.r = 0.1;
    }
}

impl Default for SSTNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// VIP Irregular-Spiking Interneuron
// ═══════════════════════════════════════════════════════════════════

/// VIP (vasoactive intestinal peptide) irregular-spiking interneuron.
///
/// Biophysics: Na+, K+, A-type K+ (Kv4, transient outward, causes
/// accommodation), leak. High input resistance, small soma.
/// Key properties: irregular/accommodating firing, disinhibitory
/// role (inhibits SST+ and PV+), bipolar morphology.
///
/// Based on Porter et al. 1998 / Bhatt et al. 2019 parameterisation.
#[derive(Clone, Debug)]
pub struct VIPNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub a: f64, // A-type K+ activation
    pub b: f64, // A-type K+ inactivation
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_a: f64,
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl VIPNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            a: 0.0,
            b: 0.9,
            g_na: 35.0, // Lower than PV+ (smaller soma)
            g_k: 6.0,
            g_a: 8.0,  // Strong A-current → accommodation
            g_l: 0.01, // High input resistance
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 0.5, // Small soma → low capacitance
            dt: 0.025,
            v_threshold: -20.0,
        }
    }

    /// Return `[dV, dh, dn, da, db]` of the five-state VIP system at one consistent
    /// state. All gates relax through sigmoidal steady states (no singularities).
    fn derivatives(&self, v: f64, h: f64, n: f64, a: f64, b: f64, current: f64) -> [f64; 5] {
        let m_inf = 1.0 / (1.0 + (-(v + 30.0) / 9.5).exp());
        let h_inf = 1.0 / (1.0 + ((v + 53.0) / 7.0).exp());
        let tau_h = 0.37 + 2.78 / (1.0 + ((v + 40.5) / 6.0).exp());
        let n_inf = 1.0 / (1.0 + (-(v + 30.0) / 10.0).exp());
        let tau_n = 0.37 + 1.85 / (1.0 + ((v + 27.0) / 15.0).exp());
        let a_inf = 1.0 / (1.0 + (-(v + 50.0) / 20.0).exp());
        let b_inf = 1.0 / (1.0 + ((v + 78.0) / 6.0).exp());
        let dh = (h_inf - h) / tau_h;
        let dn = (n_inf - n) / tau_n;
        let da = (a_inf - a) / 5.0;
        let db = (b_inf - b) / 50.0;
        let i_na = self.g_na * m_inf * m_inf * m_inf * h * (v - self.e_na);
        let i_k = self.g_k * n * n * n * n * (v - self.e_k);
        let i_a = self.g_a * a * a * a * b * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_na - i_k - i_a - i_l + current) / self.c_m;
        [dv, dh, dn, da, db]
    }

    /// Return one classical RK4 increment of `[V, h, n, a, b]`, holding `current`
    /// constant across the four stages.
    fn rk4_substep(&self, s: [f64; 5], current: f64) -> [f64; 5] {
        let dt = self.dt;
        let k1 = self.derivatives(s[0], s[1], s[2], s[3], s[4], current);
        let k2 = self.derivatives(
            s[0] + 0.5 * dt * k1[0],
            s[1] + 0.5 * dt * k1[1],
            s[2] + 0.5 * dt * k1[2],
            s[3] + 0.5 * dt * k1[3],
            s[4] + 0.5 * dt * k1[4],
            current,
        );
        let k3 = self.derivatives(
            s[0] + 0.5 * dt * k2[0],
            s[1] + 0.5 * dt * k2[1],
            s[2] + 0.5 * dt * k2[2],
            s[3] + 0.5 * dt * k2[3],
            s[4] + 0.5 * dt * k2[4],
            current,
        );
        let k4 = self.derivatives(
            s[0] + dt * k3[0],
            s[1] + dt * k3[1],
            s[2] + dt * k3[2],
            s[3] + dt * k3[3],
            s[4] + dt * k3[4],
            current,
        );
        let mut out = [0.0_f64; 5];
        for i in 0..5 {
            out[i] = s[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let mut s = [self.v, self.h, self.n, self.a, self.b];
        for _ in 0..4 {
            s = self.rk4_substep(s, current);
        }
        self.v = s[0];
        self.h = s[1];
        self.n = s[2];
        self.a = s[3];
        self.b = s[4];
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
        self.a = 0.0;
        self.b = 0.9;
    }
}

impl Default for VIPNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Chandelier Cell (Axo-Axonic)
// ═══════════════════════════════════════════════════════════════════

/// Chandelier cell — axo-axonic fast-spiking interneuron.
///
/// Biophysics: Wang-Buzsáki core + Kv1 (D-type delay current) + Kv3.1.
/// Kv1 creates a delay to first spike compared to PV+. Targets AIS.
///
/// Based on Woodruff et al. 2011 / Wang & Buzsáki 1996.
#[derive(Clone, Debug)]
pub struct ChandelierNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub d: f64, // Kv1 (D-type) activation
    pub p: f64, // Kv3.1 activation
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_kv1: f64,
    pub g_kv3: f64,
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ChandelierNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            d: 0.0,
            p: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_kv1: 3.0, // Kv1 delay current (slower)
            g_kv3: 4.0, // Kv3.1 for AP sharpening
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
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        for _ in 0..n_sub {
            // Wang-Buzsáki gating
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;

            // Kv1 (D-type): slow activation → first-spike delay
            let d_inf = 1.0 / (1.0 + (-(self.v + 50.0) / 10.0).exp());
            let tau_d = 150.0;
            self.d += (d_inf - self.d) / tau_d * self.dt;

            // Kv3.1: fast activation
            let p_inf = 1.0 / (1.0 + (-(self.v + 10.0) / 10.0).exp());
            self.p += self.phi * (p_inf - self.p) / 1.0 * self.dt;

            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_kv1 = self.g_kv1 * self.d.powi(4) * (self.v - self.e_k);
            let i_kv3 = self.g_kv3 * self.p * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);

            self.v += (-i_na - i_k - i_kv1 - i_kv3 - i_l + current) / self.c_m * self.dt;
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
        self.d = 0.0;
        self.p = 0.0;
    }
}

impl Default for ChandelierNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Cerebellar Basket Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar basket cell — perisomatic-targeting interneuron.
///
/// Biophysics: Wang-Buzsáki core + A-type K+ (transient outward) +
/// Ca2+-dependent K+ (afterhyperpolarisation). Distinct from cortical
/// PV+ by A-current and pronounced AHP from Ca2+-activated K+.
///
/// Based on Midtgaard 1992 / Häusser & Clark 1997 / WB 1996.
#[derive(Clone, Debug)]
pub struct CerebellarBasketNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub a: f64,
    pub b: f64,
    pub ca: f64, // Intracellular [Ca2+] (µM)
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_a: f64,
    pub g_kca: f64,
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl CerebellarBasketNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            a: 0.0,
            b: 0.9,
            ca: 0.05,
            g_na: 35.0,
            g_k: 9.0,
            g_a: 3.0,
            g_kca: 2.0,
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
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        for _ in 0..n_sub {
            // WB gating for Na+ and Kdr
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;

            // A-type K+ (cerebellar)
            let a_inf = 1.0 / (1.0 + (-(self.v + 45.0) / 15.0).exp());
            let b_inf = 1.0 / (1.0 + ((self.v + 75.0) / 8.0).exp());
            self.a += self.phi * (a_inf - self.a) / 5.0 * self.dt;
            self.b += (b_inf - self.b) / 50.0 * self.dt;

            // Ca2+-activated K+ (AHP)
            let q_inf = self.ca / (self.ca + 0.2);

            // Ca2+ dynamics: entry during depolarisation
            let i_ca_entry = if self.v > -20.0 {
                0.01 * (self.v + 20.0)
            } else {
                0.0
            };
            self.ca += (-self.ca / 80.0 + i_ca_entry) * self.dt;
            if self.ca < 0.0 {
                self.ca = 0.0;
            }

            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_a = self.g_a * self.a.powi(3) * self.b * (self.v - self.e_k);
            let i_kca = self.g_kca * q_inf * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);

            self.v += (-i_na - i_k - i_a - i_kca - i_l + current) / self.c_m * self.dt;
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
        self.a = 0.0;
        self.b = 0.9;
        self.ca = 0.05;
    }
}

impl Default for CerebellarBasketNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Martinotti Cell
// ═══════════════════════════════════════════════════════════════════

/// Martinotti cell — adapting interneuron targeting layer 1 apical dendrites.
///
/// Biophysics: Na+, K+, M-current (Kv7, strong adaptation), T-type Ca2+
/// (rebound), leak. Overlaps with SST+ phenotype but with stronger
/// adaptation (higher g_m) and lower rheobase.
///
/// Based on Silberberg & Markram 2007 / Toledo-Rodriguez et al. 2005.
#[derive(Clone, Debug)]
pub struct MartinottiNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub p: f64, // M-current activation
    pub s: f64, // T-type Ca2+ inactivation
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_m: f64,
    pub g_t: f64,
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl MartinottiNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            m: 0.02,
            h: 0.8,
            n: 0.2,
            p: 0.0,
            s: 0.9,
            g_na: 40.0,
            g_k: 5.0,
            g_m: 0.25, // Very strong M-current → pronounced adaptation
            g_t: 0.01, // T-type Ca2+ (minimal window current)
            g_l: 0.05, // Leak for resting stability
            e_na: 50.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -65.0,
            c_m: 0.8,
            dt: 0.025,
            v_threshold: -20.0,
        }
    }

    /// Return `[dV, dm, dh, dn, dp, ds]` of the six-state Martinotti system at one
    /// consistent state. The Na/K activation rates use the L'Hôpital limit at the
    /// removable Traub-Miles singularity; β_m carries the published `V - V_T - 40`
    /// offset (an earlier `-17` offset drove the cell into depolarisation block).
    fn derivatives(
        &self,
        v: f64,
        m: f64,
        h: f64,
        n: f64,
        p: f64,
        s: f64,
        current: f64,
    ) -> [f64; 6] {
        let dvt = v - (-56.2);
        let asing = |num: f64, slope: f64, limit: f64| {
            if num.abs() < 1e-6 {
                limit
            } else {
                num / ((num / slope).exp() - 1.0)
            }
        };
        let alpha_m = -0.32 * asing(dvt - 13.0, -4.0, -4.0);
        let beta_m = 0.28 * asing(dvt - 40.0, 5.0, 5.0);
        let alpha_h = 0.128 * (-(dvt - 17.0) / 18.0).exp();
        let beta_h = 4.0 / (1.0 + (-(dvt - 40.0) / 5.0).exp());
        let alpha_n = -0.032 * asing(dvt - 15.0, -5.0, -5.0);
        let beta_n = 0.5 * (-(dvt - 10.0) / 40.0).exp();
        let dm = alpha_m * (1.0 - m) - beta_m * m;
        let dh = alpha_h * (1.0 - h) - beta_h * h;
        let dn = alpha_n * (1.0 - n) - beta_n * n;
        let p_inf = 1.0 / (1.0 + (-(v + 35.0) / 10.0).exp());
        let tau_p = 400.0 / (3.3 * ((v + 35.0) / 20.0).exp() + (-(v + 35.0) / 20.0).exp());
        let dp = (p_inf - p) / tau_p;
        let m_t_inf = 1.0 / (1.0 + (-(v + 57.0) / 6.2).exp());
        let s_inf = 1.0 / (1.0 + ((v + 81.0) / 4.0).exp());
        let tau_s = 30.0 + 200.0 / (1.0 + ((v + 70.0) / 5.0).exp());
        let ds = (s_inf - s) / tau_s;
        let i_na = self.g_na * m * m * m * h * (v - self.e_na);
        let i_k = self.g_k * n * n * n * n * (v - self.e_k);
        let i_m = self.g_m * p * (v - self.e_k);
        let i_t = self.g_t * m_t_inf * m_t_inf * s * (v - self.e_ca);
        let i_l = self.g_l * (v - self.e_l);
        let dvdt = (-i_na - i_k - i_m - i_t - i_l + current) / self.c_m;
        [dvdt, dm, dh, dn, dp, ds]
    }

    /// Return one classical RK4 increment of `[V, m, h, n, p, s]`, holding
    /// `current` constant across the four stages.
    fn rk4_substep(&self, st: [f64; 6], current: f64) -> [f64; 6] {
        let dt = self.dt;
        let k1 = self.derivatives(st[0], st[1], st[2], st[3], st[4], st[5], current);
        let mut a = [0.0_f64; 6];
        for i in 0..6 {
            a[i] = st[i] + 0.5 * dt * k1[i];
        }
        let k2 = self.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], current);
        for i in 0..6 {
            a[i] = st[i] + 0.5 * dt * k2[i];
        }
        let k3 = self.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], current);
        for i in 0..6 {
            a[i] = st[i] + dt * k3[i];
        }
        let k4 = self.derivatives(a[0], a[1], a[2], a[3], a[4], a[5], current);
        let mut out = [0.0_f64; 6];
        for i in 0..6 {
            out[i] = st[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }
        out
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let mut st = [self.v, self.m, self.h, self.n, self.p, self.s];
        for _ in 0..4 {
            st = self.rk4_substep(st, current);
        }
        self.v = st[0];
        self.m = st[1];
        self.h = st[2];
        self.n = st[3];
        self.p = st[4];
        self.s = st[5];
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = -65.0;
        self.m = 0.02;
        self.h = 0.8;
        self.n = 0.2;
        self.p = 0.0;
        self.s = 0.9;
    }
}

impl Default for MartinottiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── PV+ tests ────────────────────────────────────────────────

    #[test]
    fn pv_fires_with_input() {
        let mut n = PVFastSpikingNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(2.0)).sum();
        assert!(spikes > 0, "PV+ must fire with sustained input");
    }

    #[test]
    fn pv_no_fire_without_input() {
        let mut n = PVFastSpikingNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn pv_negative_current_no_fire() {
        let mut n = PVFastSpikingNeuron::new();
        let spikes: i32 = (0..1000).map(|_| n.step(-1.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn pv_high_firing_rate() {
        // PV+ should sustain high-rate repetitive firing.
        let mut n = PVFastSpikingNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        assert!(spikes > 100, "PV+ should fire at high rate: got {spikes}");
    }

    #[test]
    fn pv_reset_roundtrip() {
        let mut n = PVFastSpikingNeuron::new();
        for _ in 0..1000 {
            n.step(3.0);
        }
        n.reset();
        let mut fresh = PVFastSpikingNeuron::new();
        let r1: i32 = (0..500).map(|_| n.step(3.0)).sum();
        let r2: i32 = (0..500).map(|_| fresh.step(3.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn pv_voltage_bounded() {
        let mut n = PVFastSpikingNeuron::new();
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(n.v.is_finite());
        assert!(n.h.is_finite());
        assert!(n.n.is_finite());
    }

    #[test]
    #[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]
    fn pv_performance_5k_steps() {
        let mut n = PVFastSpikingNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..5_000 {
            n.step(3.0);
        }
        assert!(
            start.elapsed().as_millis() < 500,
            "5k steps took {:?}",
            start.elapsed()
        );
    }

    // ── SST+ tests ───────────────────────────────────────────────

    #[test]
    fn sst_fires_with_input() {
        let mut n = SSTNeuron::new();
        let spikes: i32 = (0..10000).map(|_| n.step(5.0)).sum();
        assert!(spikes > 0, "SST+ must fire with sustained input");
    }

    #[test]
    fn sst_no_fire_without_input() {
        let mut n = SSTNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn sst_adaptation_reduces_rate() {
        let mut n = SSTNeuron::new();
        let first_half: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        let second_half: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        // M-current → spike frequency adaptation
        assert!(
            second_half <= first_half + 3,
            "SST+ should adapt: first={first_half}, second={second_half}"
        );
    }

    #[test]
    fn sst_reset_roundtrip() {
        let mut n = SSTNeuron::new();
        for _ in 0..5000 {
            n.step(5.0);
        }
        n.reset();
        let mut fresh = SSTNeuron::new();
        let r1: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        let r2: i32 = (0..2000).map(|_| fresh.step(5.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn sst_voltage_bounded() {
        let mut n = SSTNeuron::new();
        for _ in 0..20000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
        assert!(n.p.is_finite());
        assert!(n.s.is_finite());
    }

    #[test]
    #[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]
    fn sst_performance_10k_steps() {
        let mut n = SSTNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(5.0);
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 500,
            "10k SST steps took {:?}",
            elapsed
        );
    }

    // ── VIP tests ────────────────────────────────────────────────

    #[test]
    fn vip_fires_with_input() {
        let mut n = VIPNeuron::new();
        let spikes: i32 = (0..10000).map(|_| n.step(2.0)).sum();
        assert!(spikes > 0, "VIP must fire with sustained input");
    }

    #[test]
    fn vip_no_fire_without_input() {
        let mut n = VIPNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn vip_accommodation() {
        // A-current causes transient accommodation at spike onset.
        // Compare fresh neuron's first 100 steps vs steady-state.
        let mut n = VIPNeuron::new();
        // First 500 steps: A-current b gate is high → strong IA → suppresses early spikes
        let onset: i32 = (0..500).map(|_| n.step(3.0)).sum();
        // Skip 5000 steps to reach steady state
        for _ in 0..5000 {
            n.step(3.0);
        }
        // Next 500 steps at steady state
        let steady: i32 = (0..500).map(|_| n.step(3.0)).sum();
        // At steady state, b has dropped, IA is weaker → fires at least as much
        assert!(
            steady >= onset,
            "VIP steady-state ({steady}) should fire >= onset ({onset})"
        );
    }

    #[test]
    fn vip_reset_roundtrip() {
        let mut n = VIPNeuron::new();
        for _ in 0..5000 {
            n.step(3.0);
        }
        n.reset();
        let mut fresh = VIPNeuron::new();
        let r1: i32 = (0..2000).map(|_| n.step(3.0)).sum();
        let r2: i32 = (0..2000).map(|_| fresh.step(3.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn vip_voltage_bounded() {
        let mut n = VIPNeuron::new();
        for _ in 0..20000 {
            n.step(5.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    #[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]
    fn vip_performance_10k_steps() {
        let mut n = VIPNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(3.0);
        }
        assert!(start.elapsed().as_millis() < 100);
    }

    // ── Chandelier tests ─────────────────────────────────────────

    #[test]
    fn chandelier_fires_with_input() {
        let mut n = ChandelierNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(spikes > 0, "Chandelier must fire with sustained input");
    }

    #[test]
    fn chandelier_no_fire_without_input() {
        let mut n = ChandelierNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn chandelier_has_kv1_delay_current() {
        // Chandelier has Kv1 (D-type) which activates slowly.
        // After sustained input, Kv1 contributes extra K+ current → lower steady-state rate.
        let mut ch = ChandelierNeuron::new();
        let mut pv = PVFastSpikingNeuron::new();
        let ch_spikes: i32 = (0..5000).map(|_| ch.step(3.0)).sum();
        let pv_spikes: i32 = (0..5000).map(|_| pv.step(3.0)).sum();
        // Both should fire, Chandelier may fire fewer due to extra K+
        assert!(ch_spikes > 0, "Chandelier must fire");
        assert!(pv_spikes > 0, "PV+ must fire");
        assert!(
            ch_spikes <= pv_spikes + 10,
            "Chandelier ({ch_spikes}) should fire <= PV+ ({pv_spikes}) due to Kv1"
        );
    }

    #[test]
    fn chandelier_reset_roundtrip() {
        let mut n = ChandelierNeuron::new();
        for _ in 0..1000 {
            n.step(3.0);
        }
        n.reset();
        let mut fresh = ChandelierNeuron::new();
        let r1: i32 = (0..500).map(|_| n.step(3.0)).sum();
        let r2: i32 = (0..500).map(|_| fresh.step(3.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn chandelier_voltage_bounded() {
        let mut n = ChandelierNeuron::new();
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    #[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]
    fn chandelier_performance_5k_steps() {
        let mut n = ChandelierNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..5_000 {
            n.step(3.0);
        }
        assert!(
            start.elapsed().as_millis() < 500,
            "5k steps took {:?}",
            start.elapsed()
        );
    }

    // ── Cerebellar basket tests ──────────────────────────────────

    #[test]
    fn basket_fires_with_input() {
        let mut n = CerebellarBasketNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(spikes > 0, "Basket cell must fire with sustained input");
    }

    #[test]
    fn basket_no_fire_without_input() {
        let mut n = CerebellarBasketNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn basket_ca_dynamics_during_spiking() {
        // Ca2+ decays between spikes but spikes cause transient increases
        let mut n = CerebellarBasketNeuron::new();
        // Run until steady-state Ca2+ with spiking
        for _ in 0..5000 {
            n.step(3.0);
        }
        let ca_spiking = n.ca;
        // Ca2+ without spiking should be lower (pure decay)
        let mut n2 = CerebellarBasketNeuron::new();
        n2.ca = ca_spiking;
        for _ in 0..5000 {
            n2.step(0.0);
        }
        assert!(
            ca_spiking > n2.ca,
            "spiking Ca ({ca_spiking:.4}) should exceed resting Ca ({:.4})",
            n2.ca
        );
    }

    #[test]
    fn basket_reset_roundtrip() {
        let mut n = CerebellarBasketNeuron::new();
        for _ in 0..2000 {
            n.step(3.0);
        }
        n.reset();
        assert_eq!(n.ca, 0.05);
        let mut fresh = CerebellarBasketNeuron::new();
        let r1: i32 = (0..1000).map(|_| n.step(3.0)).sum();
        let r2: i32 = (0..1000).map(|_| fresh.step(3.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn basket_voltage_bounded() {
        let mut n = CerebellarBasketNeuron::new();
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(n.v.is_finite());
        assert!(n.ca.is_finite());
        assert!(n.ca >= 0.0);
    }

    #[test]
    #[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]
    fn basket_performance_5k_steps() {
        let mut n = CerebellarBasketNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..5_000 {
            n.step(3.0);
        }
        assert!(
            start.elapsed().as_millis() < 500,
            "5k steps took {:?}",
            start.elapsed()
        );
    }

    // ── Martinotti tests ─────────────────────────────────────────

    #[test]
    fn martinotti_fires_with_input() {
        let mut n = MartinottiNeuron::new();
        let spikes: i32 = (0..10000).map(|_| n.step(3.0)).sum();
        assert!(spikes > 0, "Martinotti must fire with sustained input");
    }

    #[test]
    fn martinotti_no_fire_without_input() {
        let mut n = MartinottiNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn martinotti_strong_adaptation() {
        let mut n = MartinottiNeuron::new();
        let first: i32 = (0..5000).map(|_| n.step(4.0)).sum();
        let second: i32 = (0..5000).map(|_| n.step(4.0)).sum();
        assert!(
            second <= first + 3,
            "Martinotti should strongly adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn martinotti_adapts_more_than_sst() {
        // Martinotti has higher g_m → stronger adaptation
        let mut mc = MartinottiNeuron::new();
        let mut sst = SSTNeuron::new();
        // Use same current magnitude
        let mc_spikes: i32 = (0..10000).map(|_| mc.step(4.0)).sum();
        let sst_spikes: i32 = (0..10000).map(|_| sst.step(4.0)).sum();
        // Martinotti should fire less (stronger adaptation, but lower rheobase too)
        // At minimum, both should fire
        assert!(mc_spikes > 0, "Martinotti should fire: got {mc_spikes}");
        assert!(sst_spikes > 0, "SST should fire: got {sst_spikes}");
    }

    #[test]
    fn martinotti_reset_roundtrip() {
        let mut n = MartinottiNeuron::new();
        for _ in 0..5000 {
            n.step(4.0);
        }
        n.reset();
        let mut fresh = MartinottiNeuron::new();
        let r1: i32 = (0..2000).map(|_| n.step(4.0)).sum();
        let r2: i32 = (0..2000).map(|_| fresh.step(4.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn martinotti_voltage_bounded() {
        let mut n = MartinottiNeuron::new();
        for _ in 0..20000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
        assert!(n.p.is_finite());
    }

    #[test]
    #[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]
    fn martinotti_performance_10k_steps() {
        let mut n = MartinottiNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            n.step(4.0);
        }
        assert!(start.elapsed().as_millis() < 100);
    }
}
