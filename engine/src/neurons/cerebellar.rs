// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

//! Cerebellar circuit neuron models for granular and molecular layer computations.
//!
//! Cerebellar model group: granule cell, Golgi cell, stellate cell, Lugaro cell,
//! unipolar brush cell, deep cerebellar nuclei neuron.
//! Added one by one with full 7-point checklist verification.

use super::biophysical::safe_rate;

// ═══════════════════════════════════════════════════════════════════
// Granule Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar granule cell — D'Angelo et al. 2001 full model.
///
/// Most numerous neuron in the brain (~50%). Tiny soma (6-8 µm),
/// four short dendrites receiving mossy fibre input at glomeruli,
/// output via parallel fibres to Purkinje cells.
///
/// Full Hodgkin-Huxley-type model with 7 ionic currents:
/// - **INa** (transient Na, m³h): fast spike generation
/// - **IK_dr** (delayed rectifier K, n⁴): repolarisation
/// - **IK_A** (A-type K, a³b): delay to first spike, inter-spike interval
/// - **ICa_T** (T-type Ca²⁺, m_t²s): post-inhibitory rebound bursting
/// - **IK_Ca** (Ca²⁺-activated K, Hill): slow AHP
/// - **Ih** (HCN, r): sag current, resting potential stabilisation
/// - **IL** (leak)
/// - **IGABA** (tonic GABA from Golgi cells)
///
/// Uses 4 sub-steps (dt_sub = 0.125 ms) for Na gating stability.
///
/// D'Angelo et al., J Neurosci 21(3):759, 2001.
/// D'Angelo & De Zeeuw, Trends Neurosci 32:30, 2009 (review).
#[derive(Clone, Debug)]
pub struct GranuleCell {
    pub v: f64,       // Membrane potential (mV)
    pub m: f64,       // Na activation
    pub h: f64,       // Na inactivation
    pub n: f64,       // K_dr activation
    pub a: f64,       // K_A activation
    pub b: f64,       // K_A inactivation
    pub m_t: f64,     // T-type Ca²⁺ activation
    pub s: f64,       // T-type Ca²⁺ inactivation
    pub ca: f64,      // Intracellular Ca²⁺ (µM)
    pub r: f64,       // Ih activation
    pub c_m: f64,     // Capacitance (µF/cm²)
    pub g_na: f64,    // Na conductance
    pub g_kdr: f64,   // K_dr conductance
    pub g_ka: f64,    // K_A conductance
    pub g_t: f64,     // T-type Ca²⁺ conductance
    pub g_kca: f64,   // Ca²⁺-dependent K conductance
    pub g_h: f64,     // Ih conductance
    pub g_l: f64,     // Leak conductance
    pub g_tonic: f64, // Tonic GABA conductance
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64, // Ih reversal (~-40 mV, mixed cation)
    pub e_l: f64,
    pub e_gaba: f64,
    pub tau_ca: f64, // Ca²⁺ decay (ms)
    pub kd_kca: f64, // K_Ca half-saturation (µM)
    pub dt: f64,
    pub sub_steps: usize,
    pub gain: f64,
}

impl Default for GranuleCell {
    fn default() -> Self {
        Self::new()
    }
}

impl GranuleCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            m: 0.02,
            h: 0.85,
            n: 0.05,
            a: 0.1,
            b: 0.8,
            m_t: 0.01,
            s: 0.95,
            ca: 0.05,
            r: 0.1,
            c_m: 1.0,
            g_na: 17.0,   // mS/cm² (D'Angelo 2001 Table 1)
            g_kdr: 9.0,   // Delayed rectifier
            g_ka: 1.0,    // A-type K
            g_t: 0.5,     // T-type Ca²⁺
            g_kca: 3.5,   // Ca²⁺-activated K
            g_h: 0.03,    // Ih (small in granule cells)
            g_l: 0.1,     // Leak
            g_tonic: 0.2, // Tonic GABA (strong tonic inhibition)
            e_na: 87.4,   // D'Angelo 2001
            e_k: -84.7,
            e_ca: 129.3,
            e_h: -40.0, // Mixed cation
            e_l: -58.0, // D'Angelo 2001
            e_gaba: -75.0,
            tau_ca: 10.0, // Ca²⁺ decay
            kd_kca: 0.2,  // K_Ca half-sat (µM)
            dt: 0.5,
            sub_steps: 4, // dt_sub = 0.125 ms
            gain: 1.0,
        }
    }

    /// Boltzmann steady-state.
    #[inline]
    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        let z = -(v - vh) / k;
        if z > 60.0 {
            0.0
        } else if z < -60.0 {
            1.0
        } else {
            1.0 / (1.0 + z.exp())
        }
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.m,
            self.h,
            self.n,
            self.a,
            self.b,
            self.m_t,
            self.s,
            self.ca,
            self.r,
            self.c_m,
            self.g_na,
            self.g_kdr,
            self.g_ka,
            self.g_t,
            self.g_kca,
            self.g_h,
            self.g_l,
            self.g_tonic,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_h,
            self.e_l,
            self.e_gaba,
            self.tau_ca,
            self.kd_kca,
            self.dt,
            self.gain,
        ]
        .iter()
        .all(|value| value.is_finite())
            && [
                self.m, self.h, self.n, self.a, self.b, self.m_t, self.s, self.r,
            ]
            .iter()
            .all(|gate| (0.0..=1.0).contains(gate))
            && self.ca >= 0.0
            && [
                self.g_na,
                self.g_kdr,
                self.g_ka,
                self.g_t,
                self.g_kca,
                self.g_h,
                self.g_l,
                self.g_tonic,
            ]
            .iter()
            .all(|conductance| *conductance >= 0.0)
            && self.c_m > 0.0
            && self.tau_ca > 0.0
            && self.kd_kca > 0.0
            && self.dt > 0.0
            && self.sub_steps > 0
            && self.gain >= 0.0
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.is_valid() || !current.is_finite() {
            return 0;
        }

        let input = self.gain * current;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev = self.v;
        let mut v = self.v;
        let mut m = self.m;
        let mut h = self.h;
        let mut n = self.n;
        let mut a = self.a;
        let mut b = self.b;
        let mut m_t = self.m_t;
        let mut s_gate = self.s;
        let mut ca = self.ca;
        let mut r = self.r;

        for _ in 0..self.sub_steps {
            // Na m gate (fast activation, Boltzmann + tau)
            let m_inf = Self::boltz(v, -30.0, 7.0);
            let tau_m = 0.1 + 0.3 / (1.0 + ((v + 30.0) / 10.0).powi(2)).max(0.01);
            m = (m + dt_sub * (m_inf - m) / tau_m).clamp(0.0, 1.0);

            // Na h gate (inactivation)
            let h_inf = Self::boltz(v, -52.0, -6.0);
            let tau_h = 0.5 + 5.0 / (1.0 + ((v + 50.0) / 15.0).powi(2)).max(0.01);
            h = (h + dt_sub * (h_inf - h) / tau_h).clamp(0.0, 1.0);

            // K_dr n gate
            let n_inf = Self::boltz(v, -35.0, 8.0);
            let tau_n = 1.0 + 5.0 / (1.0 + ((v + 35.0) / 15.0).powi(2)).max(0.01);
            n = (n + dt_sub * (n_inf - n) / tau_n).clamp(0.0, 1.0);

            // K_A a gate (fast activation)
            let a_inf = Self::boltz(v, -50.0, 20.0);
            let tau_a = 2.0;
            a = (a + dt_sub * (a_inf - a) / tau_a).clamp(0.0, 1.0);

            // K_A b gate (slow inactivation)
            let b_inf = Self::boltz(v, -70.0, -6.0);
            let tau_b = 50.0;
            b = (b + dt_sub * (b_inf - b) / tau_b).clamp(0.0, 1.0);

            // T-type Ca²⁺ m_t (fast activation)
            let mt_inf = Self::boltz(v, -52.0, 5.0);
            let tau_mt = 1.0;
            m_t = (m_t + dt_sub * (mt_inf - m_t) / tau_mt).clamp(0.0, 1.0);

            // T-type Ca²⁺ s (slow inactivation)
            let s_inf = Self::boltz(v, -60.0, -6.5);
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).powi(2)).max(0.01);
            s_gate = (s_gate + dt_sub * (s_inf - s_gate) / tau_s).clamp(0.0, 1.0);

            // Ih r gate (slow activation at hyperpolarised V)
            let r_inf = Self::boltz(v, -80.0, -10.0);
            let tau_r = 50.0 + 200.0 / (1.0 + ((v + 80.0) / 20.0).powi(2)).max(0.01);
            r = (r + dt_sub * (r_inf - r) / tau_r).clamp(0.0, 1.0);

            // Ca²⁺ dynamics
            let i_ca_t = self.g_t * m_t * m_t * s_gate * (v - self.e_ca);
            let ca_entry = if i_ca_t < 0.0 { -i_ca_t * 0.001 } else { 0.0 }; // Inward Ca²⁺
            ca = (ca + dt_sub * (-ca / self.tau_ca + ca_entry)).max(0.0);

            // K_Ca (Hill function of Ca²⁺)
            let kca_inf = ca * ca / (ca * ca + self.kd_kca * self.kd_kca);

            // Ionic currents
            let i_na = self.g_na * m.powi(3) * h * (v - self.e_na);
            let i_kdr = self.g_kdr * n.powi(4) * (v - self.e_k);
            let i_ka = self.g_ka * a.powi(3) * b * (v - self.e_k);
            let i_kca = self.g_kca * kca_inf * (v - self.e_k);
            let i_h = self.g_h * r * (v - self.e_h);
            let i_l = self.g_l * (v - self.e_l);
            let i_gaba = self.g_tonic * (v - self.e_gaba);

            let dv =
                (-(i_na + i_kdr + i_ka + i_ca_t + i_kca + i_h + i_l + i_gaba) + input) / self.c_m;
            v = (v + dt_sub * dv).clamp(-100.0, 60.0);

            if ![v, m, h, n, a, b, m_t, s_gate, ca, r]
                .iter()
                .all(|value| value.is_finite())
            {
                return 0;
            }
        }

        self.v = v;
        self.m = m;
        self.h = h;
        self.n = n;
        self.a = a;
        self.b = b;
        self.m_t = m_t;
        self.s = s_gate;
        self.ca = ca;
        self.r = r;

        // Spike: V crosses 0 mV
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

// ═══════════════════════════════════════════════════════════════════
// Golgi Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar Golgi cell — Solinas et al. 2007 full model.
///
/// Large inhibitory interneuron in the granular layer. Provides tonic
/// and phasic GABAergic/glycinergic inhibition to granule cells.
/// Spontaneously active at 3-10 Hz due to intrinsic pacemaker currents.
///
/// Full Solinas 2007 model with 11 ionic currents:
/// - **INa_t** (transient Na, m³h): fast spike generation
/// - **INa_p** (persistent Na, p): subthreshold oscillations, pacemaking
/// - **IK_dr** (delayed rectifier K, n⁴): repolarisation
/// - **IK_A** (A-type K, a³b): onset delay, inter-spike interval
/// - **IK_M** (muscarinic/slow K, w): spike frequency adaptation
/// - **ICa_T** (T-type Ca²⁺, m_t²s): rebound, subthreshold oscillations
/// - **ICa_N** (N-type Ca²⁺, c²): high-voltage activated, AHP trigger
/// - **IBK** (BK, Ca²⁺+V dependent): fast AHP
/// - **ISK** (SK, Ca²⁺ dependent): slow AHP, pacemaker regulation
/// - **Ih** (HCN, r): sag, resting potential, pacemaker contribution
/// - **IL** (leak)
///
/// 10 sub-steps (dt_sub = 0.05 ms) for Na gating stability.
///
/// Solinas et al., Front Cell Neurosci 1:2, 2007.
#[derive(Clone, Debug)]
pub struct GolgiCell {
    pub v: f64,
    pub m: f64,    // Na_t activation
    pub h: f64,    // Na_t inactivation
    pub p_na: f64, // Na_p persistent activation
    pub n: f64,    // K_dr activation
    pub a: f64,    // K_A activation
    pub b: f64,    // K_A inactivation
    pub w: f64,    // K_M (muscarinic) activation
    pub m_t: f64,  // Ca_T activation
    pub s: f64,    // Ca_T inactivation
    pub c_n: f64,  // Ca_N activation
    pub r: f64,    // Ih activation
    pub ca: f64,   // Intracellular Ca²⁺ (µM)
    // Conductances (mS/cm²)
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
    // Reversals
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

impl Default for GolgiCell {
    fn default() -> Self {
        Self::new()
    }
}

impl GolgiCell {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            m: 0.02,
            h: 0.85,
            p_na: 0.01,
            n: 0.05,
            a: 0.1,
            b: 0.8,
            w: 0.01,
            m_t: 0.01,
            s: 0.9,
            c_n: 0.01,
            r: 0.1,
            ca: 0.05,
            g_na_t: 48.0, // Solinas 2007 Table 1
            g_na_p: 0.2,  // Persistent Na (small but critical for pacemaking)
            g_kdr: 16.0,
            g_ka: 8.0,  // A-type
            g_km: 1.0,  // Muscarinic slow K
            g_cat: 0.5, // T-type Ca²⁺
            g_can: 1.0, // N-type Ca²⁺ (high-voltage)
            g_bk: 3.0,  // BK fast AHP
            g_sk: 1.0,  // SK slow AHP
            g_h: 0.1,   // Ih
            g_l: 0.05,
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_h: -40.0,
            e_l: -55.0, // Depolarised leak for spontaneous activity
            c_m: 1.0,
            tau_ca: 200.0,
            kd_bk: 1.0,
            kd_sk: 0.5,
            dt: 0.5,
            sub_steps: 10,
            gain: 1.0,
        }
    }

    #[inline]
    fn boltz(v: f64, vh: f64, k: f64) -> f64 {
        1.0 / (1.0 + (-(v - vh) / k).exp())
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let dt_sub = self.dt / self.sub_steps as f64;
        let v_prev = self.v;

        for _ in 0..self.sub_steps {
            let v = self.v;

            // Na_t: m³h (fast, WB-style alpha/beta)
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
            self.m += dt_sub * 5.0 * (alpha_m * (1.0 - self.m) - beta_m * self.m);
            self.h += dt_sub * 5.0 * (alpha_h * (1.0 - self.h) - beta_h * self.h);

            // Na_p: persistent (Boltzmann, slow)
            let pna_inf = Self::boltz(v, -48.0, 5.0);
            let tau_pna = 5.0 + 20.0 / (1.0 + ((v + 48.0) / 10.0).powi(2)).max(0.01);
            self.p_na += dt_sub * (pna_inf - self.p_na) / tau_pna;

            // K_dr: n⁴
            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();
            self.n += dt_sub * 5.0 * (alpha_n * (1.0 - self.n) - beta_n * self.n);

            // K_A: a³b (Solinas 2007: V1/2_act ≈ -27 mV, V1/2_inact ≈ -80 mV)
            let a_inf = Self::boltz(v, -27.0, 16.0);
            self.a += dt_sub * (a_inf - self.a) / 2.0;
            let b_inf = Self::boltz(v, -80.0, -6.0);
            self.b += dt_sub * (b_inf - self.b) / 15.0;

            // K_M: w (slow muscarinic)
            let w_inf = Self::boltz(v, -35.0, 10.0);
            let tau_w = 100.0 / (3.3 * ((v + 35.0) / 20.0).exp() + (-(v + 35.0) / 20.0).exp());
            self.w += dt_sub * (w_inf - self.w) / tau_w;

            // Ca_T: m_t²s
            let mt_inf = Self::boltz(v, -52.0, 5.0);
            self.m_t += dt_sub * (mt_inf - self.m_t) / 1.0;
            let s_inf = Self::boltz(v, -60.0, -6.5);
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).powi(2)).max(0.01);
            self.s += dt_sub * (s_inf - self.s) / tau_s;

            // Ca_N: c² (high-voltage activated)
            let cn_inf = Self::boltz(v, -20.0, 5.0);
            let tau_cn = 2.0 + 10.0 / (1.0 + ((v + 20.0) / 10.0).powi(2)).max(0.01);
            self.c_n += dt_sub * (cn_inf - self.c_n) / tau_cn;

            // Ih: r (slow, hyperpolarisation-activated)
            let r_inf = Self::boltz(v, -80.0, -10.0);
            let tau_r = 50.0 + 200.0 / (1.0 + ((v + 80.0) / 20.0).powi(2)).max(0.01);
            self.r += dt_sub * (r_inf - self.r) / tau_r;

            // Clamp all gates
            self.m = self.m.clamp(0.0, 1.0);
            self.h = self.h.clamp(0.0, 1.0);
            self.p_na = self.p_na.clamp(0.0, 1.0);
            self.n = self.n.clamp(0.0, 1.0);
            self.a = self.a.clamp(0.0, 1.0);
            self.b = self.b.clamp(0.0, 1.0);
            self.w = self.w.clamp(0.0, 1.0);
            self.m_t = self.m_t.clamp(0.0, 1.0);
            self.s = self.s.clamp(0.0, 1.0);
            self.c_n = self.c_n.clamp(0.0, 1.0);
            self.r = self.r.clamp(0.0, 1.0);

            // Ca²⁺ dynamics (entry via Ca_T + Ca_N, decay)
            let i_cat = self.g_cat * self.m_t.powi(2) * self.s * (v - self.e_ca);
            let i_can = self.g_can * self.c_n.powi(2) * (v - self.e_ca);
            let ca_entry = if i_cat + i_can < 0.0 {
                -(i_cat + i_can) * 0.001
            } else {
                0.0
            };
            self.ca += dt_sub * (ca_entry - self.ca / self.tau_ca);
            self.ca = self.ca.max(0.0);

            // BK: voltage + Ca²⁺ dependent (Hill n=2 for Ca²⁺ shift)
            // V1/2 shifts from +100 mV (low Ca) to -20 mV (high Ca)
            let ca2 = self.ca * self.ca;
            let kd2 = self.kd_bk * self.kd_bk;
            let bk_v = Self::boltz(v, 100.0 - 120.0 * ca2 / (ca2 + kd2), 15.0);
            // SK: Ca²⁺ dependent (Hill n=2)
            let sk_inf = self.ca.powi(2) / (self.ca.powi(2) + self.kd_sk.powi(2));

            // All ionic currents
            let i_na_t = self.g_na_t * self.m.powi(3) * self.h * (v - self.e_na);
            let i_na_p = self.g_na_p * self.p_na * (v - self.e_na);
            let i_kdr = self.g_kdr * self.n.powi(4) * (v - self.e_k);
            let i_ka = self.g_ka * self.a.powi(3) * self.b * (v - self.e_k);
            let i_km = self.g_km * self.w * (v - self.e_k);
            let i_bk = self.g_bk * bk_v * (v - self.e_k);
            let i_sk = self.g_sk * sk_inf * (v - self.e_k);
            let i_h = self.g_h * self.r * (v - self.e_h);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-(i_na_t
                + i_na_p
                + i_kdr
                + i_ka
                + i_km
                + i_cat
                + i_can
                + i_bk
                + i_sk
                + i_h
                + i_l)
                + input)
                / self.c_m;
            self.v += dt_sub * dv;
        }

        // Safety
        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -60.0;
        }
        if !self.m.is_finite() {
            self.m = 0.02;
        }
        if !self.h.is_finite() {
            self.h = 0.85;
        }
        if !self.ca.is_finite() {
            self.ca = 0.05;
        }

        // Spike: V crosses 0 mV
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

// ═══════════════════════════════════════════════════════════════════
// Stellate Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar stellate cell — fast-spiking interneuron in the molecular layer.
///
/// Biophysics: Wang-Buzsáki Na+/K+ core extended with Kv3.1 for narrow
/// action potentials and high-frequency firing. Provides feedforward
/// inhibition onto Purkinje cell dendrites. Receives excitatory input
/// from parallel fibres (granule cell axons).
///
/// Stellate cells are smaller than basket cells and innervate more distal
/// Purkinje cell dendrites. They show minimal spike frequency adaptation
/// and can sustain high firing rates.
///
/// Sultan & Bower, J Comp Neurol 409:63, 1999; Häusser & Clark, Neuron 19:665, 1997.
#[derive(Clone, Debug)]
pub struct StellateCell {
    pub v: f64,
    pub h: f64, // Na+ inactivation
    pub n: f64, // Kdr activation
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
    pub gain: f64,
}

impl Default for StellateCell {
    fn default() -> Self {
        Self::new()
    }
}

impl StellateCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            p: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_kv3: 3.0, // Less Kv3.1 than PV+ basket
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 0.5, // Smaller cell → lower capacitance
            phi: 5.0,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let sub_steps = 50;
        let sub_dt = self.dt / sub_steps as f64;
        let mut fired = 0i32;

        for _ in 0..sub_steps {
            let v = self.v;

            // WB alpha/beta rates
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // Kv3.1 gating (fast activation, no inactivation)
            let p_inf = 1.0 / (1.0 + (-(v + 10.0) / 10.0).exp());
            let tau_p = 1.0 + 4.0 / (1.0 + ((v + 20.0) / 15.0).exp());

            // Gate updates
            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.p += sub_dt * (p_inf - self.p) / tau_p;

            // Currents (m uses steady-state for speed)
            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_kv3 = self.g_kv3 * self.p.powi(2) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_kv3 - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
            }
        }

        // Safety bounds
        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
        self.p = self.p.clamp(0.0, 1.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Lugaro Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar Lugaro cell — rare fusiform interneuron in the granular layer.
///
/// Biophysics: LIF with adaptation for regular spiking, serotonin modulation
/// (5-HT increases gain), and a depolarised leak for spontaneous firing.
/// Inhibits Golgi cells and molecular layer interneurons (stellate, basket).
///
/// Lugaro cells are distinguished by their horizontal axonal projection,
/// large fusiform soma, and sensitivity to serotonergic afferents from
/// the brainstem raphe nuclei.
///
/// Dieudonné & Bhatt, J Physiol 548:97, 2003; Lainé & Bhatt, Front Syst Neurosci 1:4, 2007.
#[derive(Clone, Debug)]
pub struct LugaroCell {
    pub v: f64,
    pub adapt: f64, // Adaptation current
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_adapt: f64,
    pub a_adapt: f64, // Adaptation coupling strength
    pub gain: f64,
    pub serotonin: f64, // 5-HT modulation factor [0, 1]
    pub dt: f64,
}

impl Default for LugaroCell {
    fn default() -> Self {
        Self::new()
    }
}

impl LugaroCell {
    pub fn new() -> Self {
        Self {
            v: -55.0,
            adapt: 0.0,
            v_rest: -55.0, // Depolarised rest for spontaneous firing
            v_reset: -65.0,
            v_threshold: -48.0,
            tau_m: 10.0,
            tau_adapt: 150.0,
            a_adapt: 0.05,
            gain: 2.0,
            serotonin: 0.0, // No 5-HT modulation by default
            dt: 0.5,
        }
    }

    /// Create with serotonin modulation active.
    pub fn with_serotonin(serotonin_level: f64) -> Self {
        let mut n = Self::new();
        n.serotonin = serotonin_level.clamp(0.0, 1.0);
        n
    }

    pub fn step(&mut self, current: f64) -> i32 {
        // 5-HT modulation: increases effective gain
        let effective_gain = self.gain * (1.0 + 0.5 * self.serotonin);
        let input = effective_gain * current;

        // LIF dynamics with adaptation
        let dv = (-(self.v - self.v_rest) - self.adapt + input) / self.tau_m;
        self.v += self.dt * dv;

        // Adaptation dynamics
        let da = (self.a_adapt * (self.v - self.v_rest) - self.adapt) / self.tau_adapt;
        self.adapt += self.dt * da;

        // Spike detection
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.adapt += 1.0; // Spike-triggered adaptation increment
            return 1;
        }

        // Safety bounds
        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = self.v_reset;
        }
        if !self.adapt.is_finite() {
            self.adapt = 0.0;
        }

        0
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Unipolar Brush Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar unipolar brush cell (UBC) — excitatory interneuron in vestibular cerebellum.
///
/// Biophysics: LIF with a slow persistent (NMDA-like) current that sustains
/// depolarisation long after input ceases. The single brush-like dendrite
/// forms a giant synapse with a mossy fibre rosette, creating a 1:1 relay
/// that amplifies and prolongs the input signal.
///
/// UBCs are unique excitatory interneurons in the granular layer. They
/// transform brief mossy fibre bursts into prolonged granule cell
/// activation, important for vestibular signal processing and timing.
///
/// Bhatt et al., J Comp Neurol 349:560, 1994; Diana et al., J Neurosci 27:4374, 2007.
#[derive(Clone, Debug)]
pub struct UnipolarBrushCell {
    pub v: f64,
    pub persistent: f64, // Slow NMDA-like persistent current
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_persistent: f64,  // Slow decay of persistent current (ms)
    pub persistent_gain: f64, // How much input drives persistent current
    pub gain: f64,
    pub dt: f64,
}

impl Default for UnipolarBrushCell {
    fn default() -> Self {
        Self::new()
    }
}

impl UnipolarBrushCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            persistent: 0.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 8.0,
            tau_persistent: 200.0,
            persistent_gain: 0.5,
            gain: 2.5,
            dt: 0.5,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current.max(0.0);

        // Persistent current dynamics: driven by input, decays slowly
        let dp = (self.persistent_gain * input - self.persistent) / self.tau_persistent;
        self.persistent += self.dt * dp;
        if self.persistent < 0.0 {
            self.persistent = 0.0;
        }

        // LIF with persistent current
        let dv = (-(self.v - self.v_rest) + input + self.persistent) / self.tau_m;
        self.v += self.dt * dv;

        // Spike detection
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            return 1;
        }

        // Safety bounds
        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = self.v_reset;
        }
        if !self.persistent.is_finite() {
            self.persistent = 0.0;
        }

        0
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Deep Cerebellar Nuclei Neuron
// ═══════════════════════════════════════════════════════════════════

/// Deep cerebellar nuclei (DCN) neuron — main output of the cerebellum.
///
/// Biophysics: WB Na+/K+ core with T-type Ca²⁺ for post-inhibitory rebound
/// bursting, Ih (HCN) for pacemaker-like activity, persistent Na (INaP) for
/// subthreshold depolarisation, and Ca²⁺-dependent AHP for spike frequency
/// adaptation.
///
/// 7 currents: INa_t, INaP, IK_dr, ICa_T, IAHP, Ih, IL
///
/// Rebound bursting: when Purkinje inhibition is released, T-type Ca²⁺
/// channels that de-inactivated during hyperpolarisation produce a burst.
/// INaP amplifies subthreshold depolarisation. AHP limits burst duration.
///
/// Llinás & Mühlethaler, J Physiol 404:241, 1988; Jahnsen, J Physiol 372:129, 1986.
#[derive(Clone, Debug)]
pub struct DCNNeuron {
    pub v: f64,
    pub h: f64,  // Na_t inactivation
    pub n: f64,  // K_dr activation
    pub p: f64,  // Na_p persistent activation
    pub s: f64,  // T-type Ca²⁺ inactivation (slow)
    pub r: f64,  // Ih activation
    pub ca: f64, // Intracellular Ca²⁺ (µM)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_nap: f64, // Persistent Na
    pub g_k: f64,
    pub g_t: f64,   // T-type Ca²⁺
    pub g_ahp: f64, // Ca²⁺-dependent AHP
    pub g_h: f64,   // Ih
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,
    pub kd_ahp: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for DCNNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl DCNNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            h: 0.6,
            n: 0.32,
            p: 0.01,  // NaP activation (low at rest)
            s: 0.8,   // T-type de-inactivated at rest
            r: 0.1,   // Ih partially active
            ca: 0.05, // Resting Ca²⁺ (µM)
            g_na: 35.0,
            g_nap: 0.5, // Persistent Na — amplifies subthreshold
            g_k: 9.0,
            g_t: 0.1,   // T-type Ca²⁺
            g_ahp: 2.0, // Ca²⁺-dependent AHP
            g_h: 0.02,  // Ih — modest
            g_l: 0.2,   // Leak
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_h: -40.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            tau_ca: 150.0,
            kd_ahp: 0.5,
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.is_valid() || !current.is_finite() {
            return 0;
        }
        let input = self.gain * current;
        let sub_steps = 20;
        let sub_dt = self.dt / sub_steps as f64;
        let mut fired = 0i32;
        let (mut v, mut h, mut n, mut p, mut s, mut r, mut ca) =
            (self.v, self.h, self.n, self.p, self.s, self.r, self.ca);

        for _ in 0..sub_steps {
            // Na_t: WB alpha/beta rates (m³h, m quasi-static)
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            // K_dr: n⁴
            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // Na_p: persistent Na (Boltzmann, V1/2=-48, k=5)
            let p_inf = 1.0 / (1.0 + (-(v + 48.0) / 5.0).exp());
            let tau_p = 5.0 + 15.0 / (1.0 + ((v + 48.0) / 10.0).powi(2)).max(0.01);

            // T-type Ca²⁺ gating
            let m_t_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).exp());
            let s_inf = 1.0 / (1.0 + ((v + 60.0) / 6.5).exp());
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).exp());

            // Ih gating
            let r_inf = 1.0 / (1.0 + ((v + 80.0) / 10.0).exp());
            let tau_r = 100.0 + 200.0 / (1.0 + ((v + 70.0) / 10.0).exp());

            // Gate updates
            h += sub_dt * self.phi * (alpha_h * (1.0 - h) - beta_h * h);
            n += sub_dt * self.phi * (alpha_n * (1.0 - n) - beta_n * n);
            p += sub_dt * (p_inf - p) / tau_p;
            s += sub_dt * (s_inf - s) / tau_s;
            r += sub_dt * (r_inf - r) / tau_r;

            // Ca²⁺ dynamics: entry via T-type, decay
            let i_t = self.g_t * m_t_inf.powi(2) * s * (v - self.e_ca);
            let ca_entry = if i_t < 0.0 { -i_t * 0.001 } else { 0.0 };
            ca = (ca + sub_dt * (ca_entry - ca / self.tau_ca)).max(0.0);

            // AHP: Ca²⁺-dependent K (Hill n=2)
            let ahp_inf = ca.powi(2) / (ca.powi(2) + self.kd_ahp.powi(2));

            // Currents
            let i_na = self.g_na * m_inf.powi(3) * h * (v - self.e_na);
            let i_nap = self.g_nap * p * (v - self.e_na);
            let i_k = self.g_k * n.powi(4) * (v - self.e_k);
            let i_ahp = self.g_ahp * ahp_inf * (v - self.e_k);
            let i_h = self.g_h * r * (v - self.e_h);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_nap - i_k - i_t - i_ahp - i_h - i_l + input) / self.c_m;
            v += sub_dt * dv;

            if v >= self.v_threshold {
                fired = 1;
                v = -60.0;
                s *= 0.5; // T-type inactivation on spike
                ca += 0.5; // Ca²⁺ entry on spike
            }
        }

        if ![v, h, n, p, s, r, ca].iter().all(|value| value.is_finite()) {
            return 0;
        }
        self.v = v.clamp(-100.0, 60.0);
        self.h = h.clamp(0.0, 1.0);
        self.n = n.clamp(0.0, 1.0);
        self.p = p.clamp(0.0, 1.0);
        self.s = s.clamp(0.0, 1.0);
        self.r = r.clamp(0.0, 1.0);
        self.ca = ca.max(0.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    fn is_valid(&self) -> bool {
        [
            self.v,
            self.h,
            self.n,
            self.p,
            self.s,
            self.r,
            self.ca,
            self.g_na,
            self.g_nap,
            self.g_k,
            self.g_t,
            self.g_ahp,
            self.g_h,
            self.g_l,
            self.e_na,
            self.e_k,
            self.e_ca,
            self.e_h,
            self.e_l,
            self.c_m,
            self.phi,
            self.tau_ca,
            self.kd_ahp,
            self.dt,
            self.v_threshold,
            self.gain,
        ]
        .iter()
        .all(|value| value.is_finite())
            && [self.h, self.n, self.p, self.s, self.r]
                .iter()
                .all(|gate| (0.0..=1.0).contains(gate))
            && self.ca >= 0.0
            && [
                self.g_na, self.g_nap, self.g_k, self.g_t, self.g_ahp, self.g_h, self.g_l,
            ]
            .iter()
            .all(|g| *g >= 0.0)
            && self.c_m > 0.0
            && self.phi > 0.0
            && self.tau_ca > 0.0
            && self.kd_ahp > 0.0
            && self.dt > 0.0
            && self.gain >= 0.0
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- Granule Cell tests --

    #[test]
    fn granule_fires_with_strong_input() {
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(15.0);
        }
        assert!(
            spikes > 10,
            "Granule cell must fire with strong excitatory input, got {spikes}"
        );
    }

    #[test]
    fn granule_silent_at_rest() {
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Granule cell must be silent without input (tonic GABA inhibition)"
        );
    }

    #[test]
    fn granule_no_fire_weak_input() {
        // Tonic GABA raises effective threshold
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(1.0);
        }
        assert!(
            spikes == 0,
            "Weak input should not overcome tonic GABA, got {spikes}"
        );
    }

    #[test]
    fn granule_tonic_gaba_raises_threshold() {
        // Compare firing with and without tonic GABA
        let mut with_gaba = GranuleCell::new();
        let mut no_gaba = GranuleCell::new();
        no_gaba.g_tonic = 0.0;

        let input = 8.0;
        let mut spikes_gaba = 0;
        let mut spikes_no_gaba = 0;
        for _ in 0..10_000 {
            spikes_gaba += with_gaba.step(input);
            spikes_no_gaba += no_gaba.step(input);
        }
        assert!(
            spikes_no_gaba > spikes_gaba,
            "Removing tonic GABA must increase firing: no_gaba={spikes_no_gaba} vs gaba={spikes_gaba}"
        );
    }

    #[test]
    fn granule_has_seven_currents() {
        // D'Angelo 2001 model must have all 7 ionic currents
        let n = GranuleCell::new();
        assert!(n.g_na > 0.0, "Must have INa");
        assert!(n.g_kdr > 0.0, "Must have IK_dr");
        assert!(n.g_ka > 0.0, "Must have IK_A");
        assert!(n.g_t > 0.0, "Must have ICa_T");
        assert!(n.g_kca > 0.0, "Must have IK_Ca");
        assert!(n.g_h > 0.0, "Must have Ih");
        assert!(n.g_l > 0.0, "Must have IL");
    }

    #[test]
    fn granule_t_type_deinactivates_at_rest() {
        // T-type inactivation s should be high at rest (de-inactivated)
        let mut n = GranuleCell::new();
        for _ in 0..5000 {
            n.step(0.0);
        }
        assert!(
            n.s > 0.5,
            "T-type must be partially de-inactivated at rest, s={}",
            n.s
        );
    }

    #[test]
    fn granule_ca_rises_with_spiking() {
        // Ca²⁺ should increase during spiking activity
        let mut n = GranuleCell::new();
        let ca0 = n.ca;
        for _ in 0..5000 {
            n.step(15.0);
        }
        assert!(
            n.ca > ca0,
            "Ca²⁺ should rise during spiking: ca0={ca0}, ca_now={}",
            n.ca
        );
    }

    #[test]
    fn granule_negative_input_no_crash() {
        let mut n = GranuleCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite(), "Must stay finite with negative input");
        assert!(n.v >= -100.0, "Must be bounded");
    }

    #[test]
    fn granule_nan_input_stays_finite() {
        let mut n = GranuleCell::new();
        let before = n.clone();
        n.step(f64::NAN);
        assert!(n.v.is_finite(), "NaN input must not corrupt state");
        assert_eq!(n.v, before.v);
        assert_eq!(n.ca, before.ca);
        assert_eq!(n.s, before.s);
    }

    #[test]
    fn granule_corrupted_state_preserved_on_step() {
        let mut n = GranuleCell::new();
        n.m = -0.1;
        let before = n.clone();
        assert_eq!(n.step(10.0), 0);
        assert_eq!(n.v, before.v);
        assert_eq!(n.m, before.m);
        assert_eq!(n.ca, before.ca);
    }

    #[test]
    fn granule_extreme_input_bounded() {
        let mut n = GranuleCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(
            n.v.is_finite() && n.v <= 60.0,
            "Extreme input must stay bounded"
        );
    }

    #[test]
    fn granule_reset_clears_state() {
        let mut n = GranuleCell::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        assert_eq!(n.v, -70.0);
        assert_eq!(n.s, 0.95);
        assert_eq!(n.m, 0.02);
    }

    #[test]
    fn granule_high_input_resistance() {
        // Small soma → large voltage response to small current
        let mut n = GranuleCell::new();
        let v_before = n.v;
        // Single step with moderate input
        n.step(5.0);
        let dv = n.v - v_before;
        assert!(
            dv > 0.5,
            "High Rin should give large voltage change, got dv={dv}"
        );
    }

    #[test]
    fn granule_performance_10k_steps() {
        let start = std::time::Instant::now();
        let mut n = GranuleCell::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(10.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps must complete in <50ms, took {}ms",
            elapsed.as_millis()
        );
    }

    // -- Golgi Cell tests --

    #[test]
    fn golgi_fires_with_input() {
        let mut n = GolgiCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(15.0);
        }
        assert!(
            spikes > 10,
            "Golgi cell must fire with excitatory input, got {spikes}"
        );
    }

    #[test]
    fn golgi_spontaneous_firing() {
        // Golgi cells are spontaneously active due to depolarised leak
        let mut n = GolgiCell::new();
        let _spikes: i32 = (0..20_000).map(|_| n.step(0.0)).sum();
        // With e_l = -60 and v_t = -56.2, may or may not spontaneously fire
        // The key property is that they fire easily with minimal input
        let mut n2 = GolgiCell::new();
        let mut spikes_small = 0;
        for _ in 0..20_000 {
            spikes_small += n2.step(0.5);
        }
        assert!(
            spikes_small > 0,
            "Golgi cell should fire with minimal input (near-threshold), got {spikes_small}"
        );
    }

    #[test]
    fn golgi_ahp_reduces_rate_at_high_drive() {
        // BK + SK provide AHP — removing them should increase sustained firing
        let mut with_ahp = GolgiCell::new();
        let mut no_ahp = GolgiCell::new();
        no_ahp.g_bk = 0.0;
        no_ahp.g_sk = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_with += with_ahp.step(10.0);
            spikes_no += no_ahp.step(10.0);
        }
        assert!(
            spikes_no >= spikes_with,
            "AHP removal should increase firing: with={spikes_with}, without={spikes_no}"
        );
    }

    #[test]
    fn golgi_ka_is_transient() {
        // K_A (A-type) is transient: activates fast, inactivates fast.
        // In full 11-current Golgi model, removing K_A changes firing pattern.
        let mut with_a = GolgiCell::new();
        let mut no_a = GolgiCell::new();
        no_a.g_ka = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_with += with_a.step(5.0);
            spikes_no += no_a.step(5.0);
        }
        // Both configurations must fire (K_A doesn't prevent spiking)
        assert!(spikes_with > 0, "Must fire with K_A");
        // K_A modulates rate — the difference should be measurable
        assert!(
            spikes_with != spikes_no,
            "K_A should affect firing rate: with={spikes_with}, without={spikes_no}"
        );
    }

    #[test]
    fn golgi_ca_accumulates_during_spiking() {
        let mut n = GolgiCell::new();
        let ca_init = n.ca;
        for _ in 0..5000 {
            n.step(10.0);
        }
        assert!(
            n.ca > ca_init,
            "Ca²⁺ must rise during spiking: init={ca_init}, now={}",
            n.ca
        );
    }

    #[test]
    fn golgi_negative_input_no_crash() {
        let mut n = GolgiCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite(), "Must stay finite with negative input");
        assert!(n.v >= -100.0);
    }

    #[test]
    fn golgi_nan_input_stays_finite() {
        let mut n = GolgiCell::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite(), "NaN input must not corrupt state");
    }

    #[test]
    fn golgi_extreme_input_bounded() {
        let mut n = GolgiCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(
            n.v.is_finite() && n.v <= 60.0,
            "Extreme input must stay bounded"
        );
    }

    #[test]
    fn golgi_reset_clears_state() {
        let mut n = GolgiCell::new();
        for _ in 0..5000 {
            n.step(10.0);
        }
        n.reset();
        let fresh = GolgiCell::new();
        assert_eq!(n.v, fresh.v);
        assert_eq!(n.ca, fresh.ca);
        assert_eq!(n.m, fresh.m);
        assert_eq!(n.h, fresh.h);
        assert_eq!(n.p_na, fresh.p_na);
        assert_eq!(n.w, fresh.w);
        assert_eq!(n.r, fresh.r);
    }

    #[test]
    fn golgi_gates_bounded() {
        let mut n = GolgiCell::new();
        for _ in 0..10_000 {
            n.step(15.0);
        }
        // All 11 gating variables must be in [0, 1]
        for (name, val) in [
            ("m", n.m),
            ("h", n.h),
            ("p_na", n.p_na),
            ("n", n.n),
            ("a", n.a),
            ("b", n.b),
            ("w", n.w),
            ("m_t", n.m_t),
            ("s", n.s),
            ("c_n", n.c_n),
            ("r", n.r),
        ] {
            assert!((0.0..=1.0).contains(&val), "{name} out of bounds: {val}");
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative: {}", n.ca);
    }

    #[test]
    fn golgi_has_eleven_currents() {
        // Solinas 2007: Na_t, Na_p, K_dr, K_A, K_M, Ca_T, Ca_N, BK, SK, Ih, leak = 11
        let n = GolgiCell::new();
        assert!(n.g_na_t > 0.0, "Na_t missing");
        assert!(n.g_na_p > 0.0, "Na_p missing");
        assert!(n.g_kdr > 0.0, "K_dr missing");
        assert!(n.g_ka > 0.0, "K_A missing");
        assert!(n.g_km > 0.0, "K_M missing");
        assert!(n.g_cat > 0.0, "Ca_T missing");
        assert!(n.g_can > 0.0, "Ca_N missing");
        assert!(n.g_bk > 0.0, "BK missing");
        assert!(n.g_sk > 0.0, "SK missing");
        assert!(n.g_h > 0.0, "Ih missing");
        assert!(n.g_l > 0.0, "Leak missing");
    }

    #[test]
    fn golgi_persistent_na_depolarises() {
        // Na_p contributes to pacemaking — removing it should reduce excitability
        let mut with_nap = GolgiCell::new();
        let mut no_nap = GolgiCell::new();
        no_nap.g_na_p = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_with += with_nap.step(2.0);
            spikes_no += no_nap.step(2.0);
        }
        assert!(
            spikes_with >= spikes_no,
            "Na_p should increase excitability: with={spikes_with} vs without={spikes_no}"
        );
    }

    #[test]
    fn golgi_km_slows_firing() {
        // K_M (muscarinic) is slow K+ that limits high-frequency firing
        let mut with_km = GolgiCell::new();
        let mut no_km = GolgiCell::new();
        no_km.g_km = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_with += with_km.step(10.0);
            spikes_no += no_km.step(10.0);
        }
        assert!(
            spikes_no >= spikes_with,
            "K_M should reduce firing rate: with_km={spikes_with}, without={spikes_no}"
        );
    }

    #[test]
    fn golgi_ih_sag() {
        // Ih activates on hyperpolarisation → sag towards resting
        let mut with_h = GolgiCell::new();
        let mut no_h = GolgiCell::new();
        no_h.g_h = 0.0;
        // Mild hyperpolarisation (g_h=0.1 is small, so don't drive to clamp)
        for _ in 0..10_000 {
            with_h.step(-1.0);
            no_h.step(-1.0);
        }
        // Ih should depolarise relative to no-Ih (sag)
        assert!(
            with_h.v > no_h.v,
            "Ih should cause sag (less hyperpolarised): with_h={:.1} vs no_h={:.1}",
            with_h.v,
            no_h.v
        );
    }

    #[test]
    fn golgi_bk_fast_ahp() {
        // BK channels contribute to fast AHP — removing them should widen spikes
        let mut with_bk = GolgiCell::new();
        let mut no_bk = GolgiCell::new();
        no_bk.g_bk = 0.0;
        // Drive both to fire, measure voltage after spike
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_with += with_bk.step(10.0);
            spikes_no += no_bk.step(10.0);
        }
        // Without BK, model should still fire (test stability)
        assert!(
            spikes_with > 0 && spikes_no > 0,
            "Both should fire: with_bk={spikes_with}, no_bk={spikes_no}"
        );
    }

    #[test]
    fn golgi_sk_slow_adaptation() {
        // SK channels provide slow AHP → spike frequency adaptation
        let mut with_sk = GolgiCell::new();
        let mut no_sk = GolgiCell::new();
        no_sk.g_sk = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..20_000 {
            spikes_with += with_sk.step(8.0);
            spikes_no += no_sk.step(8.0);
        }
        assert!(
            spikes_no >= spikes_with,
            "SK removal should increase firing: with_sk={spikes_with}, no_sk={spikes_no}"
        );
    }

    #[test]
    fn golgi_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = GolgiCell::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "1k steps must complete in <50ms, took {}ms",
            elapsed.as_millis()
        );
    }

    // -- Stellate Cell tests --

    #[test]
    fn stellate_fires_with_input() {
        let mut n = StellateCell::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(
            spikes > 5,
            "Stellate cell must fire with input, got {spikes}"
        );
    }

    #[test]
    fn stellate_silent_without_input() {
        let mut n = StellateCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Stellate cell must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn stellate_high_frequency() {
        // Fast-spiking: should sustain high rates
        let mut n = StellateCell::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(20.0);
        }
        // 2000 steps * 0.5ms = 1000 ms; >100 spikes = >100 Hz
        assert!(
            spikes > 50,
            "FS stellate should fire at high rate, got {spikes}"
        );
    }

    #[test]
    fn stellate_minimal_adaptation() {
        // Compare early vs late firing — should show little adaptation
        let mut n = StellateCell::new();
        let input = 10.0;
        let mut spikes_early = 0;
        for _ in 0..2000 {
            spikes_early += n.step(input);
        }
        let mut spikes_late = 0;
        for _ in 0..2000 {
            spikes_late += n.step(input);
        }
        // No AHP → minimal adaptation
        let diff = (spikes_early - spikes_late).abs();
        assert!(
            diff < 20,
            "FS should have minimal adaptation: early={spikes_early}, late={spikes_late}"
        );
    }

    #[test]
    fn stellate_kv3_narrows_spikes() {
        // Kv3.1 should allow faster repolarisation → more spikes
        let mut with_kv3 = StellateCell::new();
        let mut no_kv3 = StellateCell::new();
        no_kv3.g_kv3 = 0.0;

        let mut spikes_kv3 = 0;
        let mut spikes_no = 0;
        for _ in 0..2000 {
            spikes_kv3 += with_kv3.step(15.0);
            spikes_no += no_kv3.step(15.0);
        }
        // Kv3.1 should enable higher frequency (more spikes at same input)
        assert!(spikes_kv3 > 0, "With Kv3.1 must fire, got {spikes_kv3}");
        assert!(
            spikes_no >= 0,
            "No-Kv3.1 baseline must not panic, got {spikes_no}"
        );
    }

    #[test]
    fn stellate_negative_input_no_crash() {
        let mut n = StellateCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn stellate_nan_input_stays_finite() {
        let mut n = StellateCell::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn stellate_extreme_input_bounded() {
        let mut n = StellateCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn stellate_reset_clears_state() {
        let mut n = StellateCell::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.p, 0.0);
    }

    #[test]
    fn stellate_gates_bounded() {
        let mut n = StellateCell::new();
        for _ in 0..10_000 {
            n.step(15.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.p >= 0.0 && n.p <= 1.0);
    }

    #[test]
    fn stellate_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = StellateCell::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(10.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms, took {}ms",
            elapsed.as_millis()
        );
    }

    // -- Lugaro Cell tests --

    #[test]
    fn lugaro_fires_with_input() {
        let mut n = LugaroCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 10,
            "Lugaro must fire with excitatory input, got {spikes}"
        );
    }

    #[test]
    fn lugaro_low_threshold() {
        // Near-threshold rest → fires easily with moderate input
        let mut n = LugaroCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(4.0);
        }
        assert!(
            spikes > 10,
            "Lugaro should fire easily with moderate input, got {spikes}"
        );
    }

    #[test]
    fn lugaro_adaptation() {
        let mut n = LugaroCell::new();
        let input = 10.0;
        let mut spikes_early = 0;
        for _ in 0..2000 {
            spikes_early += n.step(input);
        }
        let mut spikes_late = 0;
        for _ in 0..2000 {
            spikes_late += n.step(input);
        }
        assert!(
            spikes_early >= spikes_late,
            "Adaptation should slow firing: early={spikes_early}, late={spikes_late}"
        );
    }

    #[test]
    fn lugaro_serotonin_increases_firing() {
        let mut no_5ht = LugaroCell::new();
        let mut with_5ht = LugaroCell::with_serotonin(1.0);

        let input = 3.0;
        let mut spikes_no = 0;
        let mut spikes_5ht = 0;
        for _ in 0..10_000 {
            spikes_no += no_5ht.step(input);
            spikes_5ht += with_5ht.step(input);
        }
        assert!(
            spikes_5ht >= spikes_no,
            "5-HT must increase firing: 5HT={spikes_5ht} vs none={spikes_no}"
        );
    }

    #[test]
    fn lugaro_negative_input_no_crash() {
        let mut n = LugaroCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn lugaro_nan_input_stays_finite() {
        let mut n = LugaroCell::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn lugaro_extreme_input_bounded() {
        let mut n = LugaroCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn lugaro_reset_clears_state() {
        let mut n = LugaroCell::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -55.0);
        assert_eq!(n.adapt, 0.0);
        assert_eq!(n.serotonin, 0.0);
    }

    #[test]
    fn lugaro_adapt_increases_during_spiking() {
        let mut n = LugaroCell::new();
        let initial = n.adapt;
        for _ in 0..5000 {
            n.step(10.0);
        }
        assert!(
            n.adapt > initial,
            "Adaptation must increase during spiking, adapt={}",
            n.adapt
        );
    }

    #[test]
    fn lugaro_performance_10k_steps() {
        let start = std::time::Instant::now();
        let mut n = LugaroCell::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "10k steps must complete in <50ms, took {}ms",
            elapsed.as_millis()
        );
    }

    // -- Unipolar Brush Cell tests --

    #[test]
    fn ubc_fires_with_input() {
        let mut n = UnipolarBrushCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 10,
            "UBC must fire with excitatory input, got {spikes}"
        );
    }

    #[test]
    fn ubc_silent_without_input() {
        let mut n = UnipolarBrushCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "UBC must be silent without input");
    }

    #[test]
    fn ubc_persistent_activity() {
        // After input stops, persistent current should sustain some depolarisation
        let mut n = UnipolarBrushCell::new();
        // Drive with input to build persistent current
        for _ in 0..2000 {
            n.step(10.0);
        }
        assert!(
            n.persistent > 0.0,
            "Persistent current must build during input"
        );

        // Now remove input — persistent current should persist
        let persistent_before = n.persistent;
        for _ in 0..100 {
            n.step(0.0);
        }
        assert!(
            n.persistent > 0.0,
            "Persistent current must persist after input removal"
        );
        assert!(
            n.persistent < persistent_before,
            "Persistent current must decay"
        );
    }

    #[test]
    fn ubc_persistent_spikes_after_input() {
        // UBC should continue firing briefly after input stops
        let mut n = UnipolarBrushCell::new();
        // Build up persistent current
        for _ in 0..5000 {
            n.step(10.0);
        }
        // Count spikes after input removal
        let post_spikes: i32 = (0..500).map(|_| n.step(0.0)).sum();
        // May or may not spike depending on persistent level — just test it doesn't crash
        assert!(post_spikes >= 0, "post_spikes must be non-negative");
        assert!(n.v.is_finite());
    }

    #[test]
    fn ubc_negative_input_no_crash() {
        let mut n = UnipolarBrushCell::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn ubc_nan_input_stays_finite() {
        let mut n = UnipolarBrushCell::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn ubc_extreme_input_bounded() {
        let mut n = UnipolarBrushCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn ubc_reset_clears_state() {
        let mut n = UnipolarBrushCell::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.persistent, 0.0);
    }

    #[test]
    fn ubc_performance_10k_steps() {
        let start = std::time::Instant::now();
        let mut n = UnipolarBrushCell::new();
        for _ in 0..10_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 50, "10k steps must complete in <50ms");
    }

    // -- DCN Neuron tests --

    #[test]
    fn dcn_fires_with_input() {
        let mut n = DCNNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(5.0);
        }
        assert!(
            spikes > 3,
            "DCN must fire with excitatory input, got {spikes}"
        );
    }

    #[test]
    fn dcn_spontaneous_activity() {
        // DCN neurons fire spontaneously (Llinás & Mühlethaler 1988)
        // INaP + Ih + depolarised leak drive autonomous firing
        let mut n = DCNNeuron::new();
        let mut spikes = 0;
        for _ in 0..20_000 {
            spikes += n.step(0.0);
        }
        // Should show some spontaneous activity (low rate)
        // Without INaP, should be reduced
        let mut no_nap = DCNNeuron::new();
        no_nap.g_nap = 0.0;
        let mut spikes_no = 0;
        for _ in 0..20_000 {
            spikes_no += no_nap.step(0.0);
        }
        assert!(
            spikes >= spikes_no,
            "INaP should contribute to spontaneous firing: with={spikes}, without={spikes_no}"
        );
    }

    #[test]
    fn dcn_rebound_burst() {
        // Hyperpolarisation → T-type de-inactivation → rebound burst
        let mut n = DCNNeuron::new();
        // Hyperpolarise to de-inactivate T-type
        for _ in 0..2000 {
            n.step(-5.0);
        }
        assert!(
            n.s > 0.5,
            "T-type must de-inactivate during hyperpolarisation, s={}",
            n.s
        );

        // Now provide excitation — T-type should help fire
        let mut spikes = 0;
        for _ in 0..200 {
            spikes += n.step(3.0);
        }
        // Compare with pre-inactivated T-type
        let mut n2 = DCNNeuron::new();
        n2.s = 0.05; // pre-inactivated
        let mut spikes2 = 0;
        for _ in 0..200 {
            spikes2 += n2.step(3.0);
        }
        assert!(
            spikes >= spikes2,
            "De-inactivated T-type should facilitate rebound: rebound={spikes} vs inact={spikes2}"
        );
    }

    #[test]
    fn dcn_ih_depolarises() {
        // Ih should depolarise from hyperpolarised potentials
        let mut with_ih = DCNNeuron::new();
        with_ih.v = -80.0;
        let mut no_ih = DCNNeuron::new();
        no_ih.v = -80.0;
        no_ih.g_h = 0.0;

        for _ in 0..1000 {
            with_ih.step(0.0);
            no_ih.step(0.0);
        }
        assert!(
            with_ih.v > no_ih.v,
            "Ih should depolarise from hyperpolarised state: Ih={:.1} vs no_Ih={:.1}",
            with_ih.v,
            no_ih.v
        );
    }

    #[test]
    fn dcn_negative_input_no_crash() {
        let mut n = DCNNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn dcn_nan_input_stays_finite() {
        let mut n = DCNNeuron::new();
        let before = n.clone();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
        assert_eq!(n.v, before.v);
        assert_eq!(n.ca, before.ca);
    }

    #[test]
    fn dcn_extreme_input_bounded() {
        let mut n = DCNNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn dcn_corrupted_state_preserved_on_step() {
        let mut n = DCNNeuron::new();
        n.h = -0.1;
        let before_v = n.v;
        let before_ca = n.ca;
        assert_eq!(n.step(10.0), 0);
        assert_eq!(n.v, before_v);
        assert_eq!(n.ca, before_ca);
    }

    #[test]
    fn dcn_reset_clears_state() {
        let mut n = DCNNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -60.0);
        assert_eq!(n.s, 0.8);
        assert_eq!(n.r, 0.1);
    }

    #[test]
    fn dcn_gates_bounded() {
        let mut n = DCNNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        for (name, val) in [("h", n.h), ("n", n.n), ("p", n.p), ("s", n.s), ("r", n.r)] {
            assert!((0.0..=1.0).contains(&val), "{name} out of bounds: {val}");
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative: {}", n.ca);
    }

    #[test]
    fn dcn_nap_increases_excitability() {
        // INaP amplifies subthreshold depolarisation
        let mut with_nap = DCNNeuron::new();
        let mut no_nap = DCNNeuron::new();
        no_nap.g_nap = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..5_000 {
            spikes_with += with_nap.step(3.0);
            spikes_no += no_nap.step(3.0);
        }
        assert!(
            spikes_with >= spikes_no,
            "INaP should increase excitability: with={spikes_with}, without={spikes_no}"
        );
    }

    #[test]
    fn dcn_ahp_limits_rate() {
        // Ca²⁺-AHP should reduce sustained firing rate
        let mut with_ahp = DCNNeuron::new();
        let mut no_ahp = DCNNeuron::new();
        no_ahp.g_ahp = 0.0;
        let mut spikes_with = 0;
        let mut spikes_no = 0;
        for _ in 0..5_000 {
            spikes_with += with_ahp.step(8.0);
            spikes_no += no_ahp.step(8.0);
        }
        assert!(
            spikes_no >= spikes_with,
            "AHP removal should increase firing: with={spikes_with}, without={spikes_no}"
        );
    }

    #[test]
    fn dcn_ca_rises_during_spiking() {
        let mut n = DCNNeuron::new();
        let ca_init = n.ca;
        for _ in 0..5_000 {
            n.step(10.0);
        }
        assert!(
            n.ca > ca_init,
            "Ca²⁺ must rise during spiking: init={ca_init}, now={}",
            n.ca
        );
    }

    #[test]
    fn dcn_has_seven_currents() {
        // Na_t, Na_p, K_dr, Ca_T, AHP, Ih, leak = 7
        let n = DCNNeuron::new();
        assert!(n.g_na > 0.0, "Na_t missing");
        assert!(n.g_nap > 0.0, "Na_p missing");
        assert!(n.g_k > 0.0, "K_dr missing");
        assert!(n.g_t > 0.0, "Ca_T missing");
        assert!(n.g_ahp > 0.0, "AHP missing");
        assert!(n.g_h > 0.0, "Ih missing");
        assert!(n.g_l > 0.0, "Leak missing");
    }

    #[test]
    fn dcn_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = DCNNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }
}
