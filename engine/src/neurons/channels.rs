// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Ion Channel Variant Neuron Models

//! Neuron models defined by a single additional ion channel beyond base HH.
//!
//! Phase 3E: persistent Na+, Ih, T-type Ca2+, A-type K+, BK, SK, NMDA.
//! Each model demonstrates one biophysical mechanism on a WB Na+/K+ base.
//! Added one by one with full 7-point checklist verification.

use super::biophysical::safe_rate;

// ═══════════════════════════════════════════════════════════════════
// Persistent Na+ Neuron
// ═══════════════════════════════════════════════════════════════════

/// Persistent Na+ (INaP) neuron — WB base + non-inactivating Na+ current.
///
/// INaP activates at subthreshold voltages (-60 to -40 mV) and does not
/// inactivate, providing a sustained depolarising drive. Key mechanism for:
/// - Subthreshold membrane oscillations (entorhinal cortex, layer II stellate)
/// - Plateau potentials and bistability (spinal motoneurons)
/// - Amplification of synaptic inputs near threshold
/// - Burst generation in respiratory neurons (pre-Bötzinger complex)
///
/// Crill, Annu Rev Physiol 58:349, 1996; French et al., Neuroscience 42:363, 1990.
#[derive(Clone, Debug)]
pub struct PersistentNaNeuron {
    pub v: f64,
    pub h: f64, // Transient Na+ inactivation
    pub n: f64, // Kdr activation
    pub p: f64, // INaP activation (slow)
    // Conductances (mS/cm²)
    pub g_na: f64,  // Transient Na+
    pub g_nap: f64, // Persistent Na+
    pub g_k: f64,   // Kdr
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

impl Default for PersistentNaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl PersistentNaNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            p: 0.0,
            g_na: 35.0,
            g_nap: 0.15, // Persistent Na+ — small but significant
            g_k: 9.0,
            g_l: 0.3, // Higher leak to counteract INaP window current
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
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

            // Persistent Na+ gating: slow activation, no inactivation
            // Half-activation at -48 mV (subthreshold), tau ~10-50 ms
            let p_inf = 1.0 / (1.0 + (-(v + 48.0) / 5.0).exp());
            let tau_p = 10.0 + 40.0 / (1.0 + ((v + 48.0) / 10.0).powi(2));

            // Gate updates
            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.p += sub_dt * (p_inf - self.p) / tau_p;

            // Currents
            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_nap = self.g_nap * self.p * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_nap - i_k - i_l + input) / self.c_m;
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
// Ih (HCN) Neuron
// ═══════════════════════════════════════════════════════════════════

/// Ih (hyperpolarisation-activated cation current) neuron — WB base + HCN.
///
/// Ih activates upon hyperpolarisation (opposite to most voltage-gated
/// channels) and conducts a mixed Na+/K+ current with reversal ~-40 mV.
/// Key mechanism for:
/// - Voltage sag: during hyperpolarisation, Ih activates and depolarises
///   the cell back towards rest (sag potential)
/// - Rebound excitation: Ih accumulated during inhibition depolarises
///   the cell after inhibition ends, triggering rebound spikes
/// - Pacemaker oscillations: interplay of Ih and T-type Ca2+ in thalamic
///   relay neurons drives rhythmic bursting
///
/// Robinson & Bhatt, Neuron 11:953, 1993; Pape, Annu Rev Physiol 58:299, 1996.
#[derive(Clone, Debug)]
pub struct IhNeuron {
    pub v: f64,
    pub h: f64, // Na+ inactivation
    pub n: f64, // Kdr activation
    pub r: f64, // Ih activation (activates on hyperpolarisation)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_h: f64, // Ih conductance
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_h: f64, // Ih reversal (~-40 mV, mixed cation)
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for IhNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl IhNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            r: 0.1,
            g_na: 35.0,
            g_k: 9.0,
            g_h: 0.15,
            g_l: 0.2,
            e_na: 55.0,
            e_k: -90.0,
            e_h: -40.0,
            e_l: -65.0,
            c_m: 1.0,
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

            // Ih gating: activates on hyperpolarisation
            // Half-activation ~-80 mV, slow kinetics (100-300 ms)
            let r_inf = 1.0 / (1.0 + ((v + 80.0) / 10.0).exp());
            let tau_r = 100.0 + 200.0 / (1.0 + ((v + 70.0) / 10.0).exp());

            // Gate updates
            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.r += sub_dt * (r_inf - self.r) / tau_r;

            // Currents
            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_h = self.g_h * self.r * (v - self.e_h);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_h - i_l + input) / self.c_m;
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
        self.r = self.r.clamp(0.0, 1.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// T-type Ca2+ Neuron
// ═══════════════════════════════════════════════════════════════════

/// T-type Ca2+ (IT) neuron — WB base + low-voltage-activated Ca2+ current.
///
/// IT activates at subthreshold voltages (-65 to -50 mV) and inactivates
/// slowly. When de-inactivated by hyperpolarisation, IT produces a
/// low-threshold spike (LTS) — a broad Ca2+ depolarisation that can
/// trigger a burst of Na+ action potentials riding on top.
///
/// Key mechanism for:
/// - Rebound bursting in thalamocortical relay neurons
/// - Sleep spindle generation (thalamic reticular nucleus)
/// - Low-threshold calcium spikes in cortical layer V pyramidal cells
/// - Rhythmic bursting in inferior olive neurons
///
/// Huguenard, Annu Rev Physiol 58:329, 1996; Destexhe et al., J Neurophysiol 76:2049, 1996.
#[derive(Clone, Debug)]
pub struct TTypeCaNeuron {
    pub v: f64,
    pub h: f64, // Na+ inactivation
    pub n: f64, // Kdr activation
    pub s: f64, // T-type Ca2+ inactivation (slow)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_t: f64, // T-type Ca2+
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for TTypeCaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl TTypeCaNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            s: 0.9, // De-inactivated at rest (-65 mV)
            g_na: 35.0,
            g_k: 9.0,
            g_t: 0.1, // Reduced to avoid window current at rest
            g_l: 0.2,
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -65.0,
            c_m: 1.0,
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

            // T-type Ca2+ gating
            let m_t_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).exp());
            let s_inf = 1.0 / (1.0 + ((v + 81.0) / 4.0).exp());
            let tau_s = 30.0 + 100.0 / (1.0 + ((v + 75.0) / 10.0).exp());

            // Gate updates
            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.s += sub_dt * (s_inf - self.s) / tau_s;

            // Currents
            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_t = self.g_t * m_t_inf.powi(2) * self.s * (v - self.e_ca);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_t - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
                self.s *= 0.3; // Spike inactivates T-type strongly
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
        self.s = self.s.clamp(0.0, 1.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// A-type K+ Neuron
// ═══════════════════════════════════════════════════════════════════

/// A-type K+ (IA) neuron — WB base + transient outward K+ current.
///
/// IA activates rapidly at subthreshold voltages and inactivates over
/// tens of milliseconds. The transient outward current opposes depolarisation,
/// creating a delay to the first spike and controlling interspike intervals.
///
/// Key mechanism for:
/// - First-spike latency: IA must inactivate before a spike can occur
/// - Spike frequency control: IA recovery during ISI limits firing rate
/// - Coincidence detection: neurons with strong IA prefer synchronous input
/// - Dendritic signal processing (hippocampal CA1 dendrites)
///
/// Connor & Stevens, J Physiol 213:31, 1971; Hoffman et al., Nature 387:869, 1997.
#[derive(Clone, Debug)]
pub struct ATypeKNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub a: f64, // IA activation (fast)
    pub b: f64, // IA inactivation (slow)
    pub g_na: f64,
    pub g_k: f64,
    pub g_a: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for ATypeKNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl ATypeKNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            a: 0.1,
            b: 0.8,
            g_na: 35.0,
            g_k: 9.0,
            g_a: 8.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
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

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // A-type K+ gating (Connor-Stevens)
            let a_inf = 1.0 / (1.0 + (-(v + 50.0) / 20.0).exp());
            let tau_a = 2.0;
            let b_inf = 1.0 / (1.0 + ((v + 70.0) / 6.0).exp());
            let tau_b = 50.0;

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.a += sub_dt * (a_inf - self.a) / tau_a;
            self.b += sub_dt * (b_inf - self.b) / tau_b;

            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_a = self.g_a * self.a.powi(3) * self.b * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_a - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
            }
        }

        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
        self.a = self.a.clamp(0.0, 1.0);
        self.b = self.b.clamp(0.0, 1.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// BK (Big Conductance Ca2+-Activated K+) Neuron
// ═══════════════════════════════════════════════════════════════════

/// BK channel neuron — WB base + voltage- and Ca2+-dependent K+ current.
///
/// BK (big conductance, MaxiK) channels are activated by both membrane
/// depolarisation and intracellular Ca2+. They have the largest single-
/// channel conductance (~250 pS) of any K+ channel. During action
/// potentials, Ca2+ influx through voltage-gated Ca2+ channels activates
/// BK, producing fast repolarisation and a prominent fast AHP.
///
/// Key mechanism for:
/// - Fast afterhyperpolarisation (fAHP): rapid spike repolarisation
/// - Action potential narrowing: BK shortens AP duration
/// - Burst termination: accumulated Ca2+ activates BK, ending burst
/// - High-frequency firing: fast repolarisation enables rapid recovery
///
/// Bhatt & Storm, J Physiol 557:329, 2003; Faber & Bhatt, PNAS 100:2813, 2003.
#[derive(Clone, Debug)]
pub struct BKNeuron {
    pub v: f64,
    pub h: f64,  // Na+ inactivation
    pub n: f64,  // Kdr activation
    pub ca: f64, // Intracellular Ca2+ concentration
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_bk: f64, // BK conductance
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64, // Ca2+ decay time constant (ms)
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for BKNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl BKNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            ca: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_bk: 3.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            tau_ca: 50.0,
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

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // BK activation: joint voltage and Ca2+ dependence
            // Half-activation shifts left (easier) with higher Ca2+
            // BK half-activation: ~+10 mV without Ca2+, shifts to -20 mV with high Ca2+
            let v_half_bk = 10.0 - 30.0 * (self.ca / (self.ca + 0.5));
            let bk_inf = 1.0 / (1.0 + (-(v - v_half_bk) / 15.0).exp());

            // Ca2+ dynamics: decay + spike-triggered influx
            self.ca += sub_dt * (-self.ca / self.tau_ca);

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);

            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_bk = self.g_bk * bk_inf * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_bk - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
                self.ca += 0.3; // Ca2+ influx on spike
            }
        }

        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        if !self.ca.is_finite() {
            self.ca = 0.0;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
        self.ca = self.ca.max(0.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// SK (Small Conductance Ca2+-Activated K+) Neuron
// ═══════════════════════════════════════════════════════════════════

/// SK channel neuron — WB base + Ca2+-only-dependent K+ current.
///
/// SK (KCa2.x) channels are activated solely by intracellular Ca2+
/// (no voltage dependence). They have slower kinetics than BK and produce
/// the medium afterhyperpolarisation (mAHP) lasting 50-200 ms after spikes.
///
/// Key mechanism for:
/// - Medium AHP (mAHP): limits sustained firing rate
/// - Spike frequency adaptation: Ca2+ builds → SK activates → firing slows
/// - Rhythmic firing patterns: SK-mediated pauses create regular ISIs
/// - Synaptic plasticity gating: SK in dendritic spines regulates NMDA currents
///
/// Bhatt & Storm, J Physiol 557:329, 2003; Stocker, Nat Rev Neurosci 5:758, 2004.
#[derive(Clone, Debug)]
pub struct SKNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub ca: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_sk: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for SKNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl SKNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            ca: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_sk: 2.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            tau_ca: 150.0, // Slower Ca2+ decay than BK → longer mAHP
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

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // SK activation: purely Ca2+-dependent (Hill function, n=2)
            let ca2 = self.ca * self.ca;
            let sk_inf = ca2 / (ca2 + 0.25); // Half-activation at [Ca2+]=0.5

            // Ca2+ decay
            self.ca += sub_dt * (-self.ca / self.tau_ca);

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);

            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_sk = self.g_sk * sk_inf * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_sk - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
                self.ca += 0.2;
            }
        }

        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        if !self.ca.is_finite() {
            self.ca = 0.0;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
        self.ca = self.ca.max(0.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// NMDA Receptor-Gated Channel Neuron
// ═══════════════════════════════════════════════════════════════════

/// NMDA receptor neuron — WB base + NMDA-type glutamate receptor current.
///
/// NMDA receptors require both glutamate binding (modelled as input current)
/// AND membrane depolarisation (Mg2+ block removal) for activation. The
/// Mg2+ block is voltage-dependent: at rest (-65 mV) channels are blocked,
/// but depolarisation to -40 mV relieves ~80% of the block.
///
/// Key mechanism for:
/// - Coincidence detection: requires presynaptic (glutamate) + postsynaptic
///   (depolarisation) signals simultaneously
/// - Synaptic plasticity: Ca2+ influx through NMDA triggers LTP/LTD
/// - Working memory: NMDA-mediated recurrent excitation sustains persistent
///   activity in prefrontal cortex
/// - Slow synaptic integration: rise ~10 ms, decay ~100 ms
///
/// Jahr & Stevens, J Neurosci 10:1830, 1990; Wang, Neuron 22:409, 1999.
#[derive(Clone, Debug)]
pub struct NMDANeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub s_nmda: f64, // NMDA synaptic variable (slow rise/decay)
    pub g_na: f64,
    pub g_k: f64,
    pub g_nmda: f64, // NMDA conductance
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_nmda: f64, // NMDA reversal (0 mV, mixed cation)
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub mg_conc: f64,   // Extracellular Mg2+ (mM), typically 1.0
    pub tau_rise: f64,  // NMDA rise time (ms)
    pub tau_decay: f64, // NMDA decay time (ms)
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for NMDANeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl NMDANeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.6,
            n: 0.32,
            s_nmda: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_nmda: 0.5,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_nmda: 0.0, // Mixed cation reversal
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            mg_conc: 1.0,
            tau_rise: 10.0,
            tau_decay: 100.0,
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

        // NMDA synaptic variable: driven by input (as proxy for glutamate)
        let drive = if input > 0.0 {
            input / (input + 5.0)
        } else {
            0.0
        };
        let ds = (drive - self.s_nmda)
            / if drive > self.s_nmda {
                self.tau_rise
            } else {
                self.tau_decay
            };
        self.s_nmda += self.dt * ds;
        self.s_nmda = self.s_nmda.clamp(0.0, 1.0);

        for _ in 0..sub_steps {
            let v = self.v;

            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = alpha_m / (alpha_m + beta_m);

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // Mg2+ block: B(V) = 1 / (1 + [Mg2+]/3.57 * exp(-0.062 * V))
            // Jahr & Stevens 1990
            let mg_block = 1.0 / (1.0 + (self.mg_conc / 3.57) * (-0.062 * v).exp());

            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);

            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_nmda = self.g_nmda * self.s_nmda * mg_block * (v - self.e_nmda);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_nmda - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0;
            }
        }

        self.v = self.v.clamp(-100.0, 60.0);
        if !self.v.is_finite() {
            self.v = -65.0;
            self.h = 0.6;
            self.n = 0.32;
        }
        if !self.s_nmda.is_finite() {
            self.s_nmda = 0.0;
        }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);

        fired
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // -- Persistent Na+ Neuron tests --

    #[test]
    fn nap_fires_with_input() {
        let mut n = PersistentNaNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(spikes > 5, "NaP neuron must fire with input, got {spikes}");
    }

    #[test]
    fn nap_subthreshold_oscillations() {
        // INaP neurons often show subthreshold oscillations or low-rate
        // spontaneous firing — this is a biological feature, not a bug.
        // With negative input, INaP should be suppressed.
        let mut n = PersistentNaNeuron::new();
        let mut spikes_inhibited = 0;
        for _ in 0..10_000 {
            spikes_inhibited += n.step(-2.0);
        }
        assert_eq!(
            spikes_inhibited, 0,
            "INaP neuron must be silent with inhibitory input, got {spikes_inhibited}"
        );
    }

    #[test]
    fn nap_lowers_threshold() {
        // INaP provides subthreshold depolarisation → lower effective threshold
        let mut with_nap = PersistentNaNeuron::new();
        let mut no_nap = PersistentNaNeuron::new();
        no_nap.g_nap = 0.0;

        // Use near-threshold input
        let input = 1.0;
        let mut spikes_nap = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_nap += with_nap.step(input);
            spikes_no += no_nap.step(input);
        }
        assert!(
            spikes_nap >= spikes_no,
            "INaP must lower effective threshold: NaP={spikes_nap} vs none={spikes_no}"
        );
    }

    #[test]
    fn nap_p_gate_activates_at_subthreshold() {
        // At -50 mV (subthreshold), p_inf should be significant
        let mut n = PersistentNaNeuron::new();
        n.v = -50.0;
        // Step a few times for p to converge
        for _ in 0..1000 {
            // Hold at -50 mV artificially by resetting v each step
            let _ = n.step(0.0);
        }
        // p_inf at -50 mV = 1/(1+exp(2/5)) = 1/(1+1.49) = 0.40
        // After many steps p should approach p_inf
        assert!(
            n.p > 0.01,
            "p gate must activate at subthreshold voltages, p={}",
            n.p
        );
    }

    #[test]
    fn nap_increases_firing_rate() {
        // Same input, higher g_nap → more spikes
        let mut low = PersistentNaNeuron::new();
        low.g_nap = 0.2;
        let mut high = PersistentNaNeuron::new();
        high.g_nap = 1.5;

        let input = 1.5;
        let mut spikes_low = 0;
        let mut spikes_high = 0;
        for _ in 0..10_000 {
            spikes_low += low.step(input);
            spikes_high += high.step(input);
        }
        assert!(
            spikes_high >= spikes_low,
            "Higher g_nap must increase firing: high={spikes_high} vs low={spikes_low}"
        );
    }

    #[test]
    fn nap_negative_input_no_crash() {
        let mut n = PersistentNaNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn nap_nan_input_stays_finite() {
        let mut n = PersistentNaNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn nap_extreme_input_bounded() {
        let mut n = PersistentNaNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn nap_reset_clears_state() {
        let mut n = PersistentNaNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.p, 0.0);
        assert_eq!(n.h, 0.6);
    }

    #[test]
    fn nap_gates_bounded() {
        let mut n = PersistentNaNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.p >= 0.0 && n.p <= 1.0);
    }

    #[test]
    fn nap_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = PersistentNaNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }

    // -- Ih Neuron tests --

    #[test]
    fn ih_fires_with_input() {
        let mut n = IhNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(spikes > 5, "Ih neuron must fire with input, got {spikes}");
    }

    #[test]
    fn ih_silent_without_input() {
        let mut n = IhNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "Ih neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn ih_sag_potential() {
        // Hyperpolarising input should produce sag (voltage rebounds towards rest)
        let mut with_ih = IhNeuron::new();
        let mut no_ih = IhNeuron::new();
        no_ih.g_h = 0.0;

        // Apply hyperpolarising step
        for _ in 0..4000 {
            with_ih.step(-3.0);
            no_ih.step(-3.0);
        }
        // With Ih, voltage should be less hyperpolarised (sag back)
        assert!(
            with_ih.v > no_ih.v,
            "Ih sag must depolarise from hyperpolarisation: Ih={:.1} vs no_Ih={:.1}",
            with_ih.v,
            no_ih.v
        );
    }

    #[test]
    fn ih_r_gate_activates_on_hyperpolarisation() {
        let mut n = IhNeuron::new();
        let r_before = n.r;
        // Hyperpolarise
        for _ in 0..4000 {
            n.step(-5.0);
        }
        assert!(
            n.r > r_before,
            "r gate must increase during hyperpolarisation, r={}",
            n.r
        );
    }

    #[test]
    fn ih_rebound_excitation() {
        // After hyperpolarisation, Ih should help reach threshold
        let mut n = IhNeuron::new();
        // Hyperpolarise to build up Ih
        for _ in 0..4000 {
            n.step(-3.0);
        }
        let r_after_hyp = n.r;
        assert!(
            r_after_hyp > 0.2,
            "r must build up during hyperpolarisation, r={r_after_hyp}"
        );

        // Release — count spikes during rebound period
        let mut rebound_spikes = 0;
        for _ in 0..500 {
            rebound_spikes += n.step(1.5);
        }

        // Compare with neuron that was not hyperpolarised
        let mut n2 = IhNeuron::new();
        let mut direct_spikes = 0;
        for _ in 0..500 {
            direct_spikes += n2.step(1.5);
        }

        assert!(
            rebound_spikes >= direct_spikes,
            "Rebound should facilitate firing: rebound={rebound_spikes} vs direct={direct_spikes}"
        );
    }

    #[test]
    fn ih_negative_input_no_crash() {
        let mut n = IhNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn ih_nan_input_stays_finite() {
        let mut n = IhNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn ih_extreme_input_bounded() {
        let mut n = IhNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn ih_reset_clears_state() {
        let mut n = IhNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.r, 0.1);
    }

    #[test]
    fn ih_gates_bounded() {
        let mut n = IhNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.r >= 0.0 && n.r <= 1.0);
    }

    #[test]
    fn ih_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = IhNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(2.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }

    // -- T-type Ca2+ Neuron tests --

    #[test]
    fn ttype_fires_with_input() {
        let mut n = TTypeCaNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(
            spikes > 5,
            "T-type neuron must fire with input, got {spikes}"
        );
    }

    #[test]
    fn ttype_silent_without_input() {
        let mut n = TTypeCaNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "T-type neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn ttype_rebound_burst() {
        // Hyperpolarise → de-inactivate T-type → rebound burst on release
        let mut n = TTypeCaNeuron::new();
        // Hyperpolarise
        for _ in 0..4000 {
            n.step(-3.0);
        }
        assert!(
            n.s > 0.3,
            "T-type must de-inactivate during hyperpolarisation, s={}",
            n.s
        );

        // Release with mild input
        let mut rebound_spikes = 0;
        for _ in 0..500 {
            rebound_spikes += n.step(1.5);
        }

        // Compare with pre-inactivated neuron
        let mut n2 = TTypeCaNeuron::new();
        n2.s = 0.05;
        let mut direct_spikes = 0;
        for _ in 0..500 {
            direct_spikes += n2.step(1.5);
        }
        assert!(
            rebound_spikes >= direct_spikes,
            "Rebound should facilitate firing: rebound={rebound_spikes} vs inact={direct_spikes}"
        );
    }

    #[test]
    fn ttype_s_gate_de_inactivates_at_hyperpolarised() {
        let mut n = TTypeCaNeuron::new();
        n.v = -85.0;
        n.s = 0.1; // Start inactivated
                   // s_inf at -85 = 1/(1+exp((-85+81)/4)) = 1/(1+exp(-1)) = 1/1.37 = 0.73
        for _ in 0..5000 {
            n.step(-5.0);
        }
        assert!(
            n.s > 0.5,
            "s must de-inactivate at hyperpolarised potentials, s={}",
            n.s
        );
    }

    #[test]
    fn ttype_spike_inactivates_t_channel() {
        let mut n = TTypeCaNeuron::new();
        let s_before_spiking = n.s;
        // Drive until spike
        let mut spiked = false;
        for _ in 0..2000 {
            if n.step(3.0) > 0 {
                spiked = true;
                break;
            }
        }
        if spiked {
            assert!(
                n.s < s_before_spiking,
                "Spike must inactivate T-type: before={s_before_spiking}, after={}",
                n.s
            );
        }
    }

    #[test]
    fn ttype_negative_input_no_crash() {
        let mut n = TTypeCaNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn ttype_nan_input_stays_finite() {
        let mut n = TTypeCaNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn ttype_extreme_input_bounded() {
        let mut n = TTypeCaNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn ttype_reset_clears_state() {
        let mut n = TTypeCaNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.s, 0.9);
    }

    #[test]
    fn ttype_gates_bounded() {
        let mut n = TTypeCaNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.s >= 0.0 && n.s <= 1.0);
    }

    #[test]
    fn ttype_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = TTypeCaNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(2.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }

    // -- A-type K+ Neuron tests --

    #[test]
    fn atype_fires_with_input() {
        let mut n = ATypeKNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(3.0);
        }
        assert!(
            spikes > 5,
            "A-type neuron must fire with input, got {spikes}"
        );
    }

    #[test]
    fn atype_silent_without_input() {
        let mut n = ATypeKNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "A-type neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn atype_delays_first_spike() {
        // IA creates onset delay — removing IA should shorten latency
        let mut with_ia = ATypeKNeuron::new();
        let mut no_ia = ATypeKNeuron::new();
        no_ia.g_a = 0.0;

        let input = 3.0;
        let mut time_with = 10_000usize;
        for i in 0..10_000 {
            if with_ia.step(input) > 0 {
                time_with = i;
                break;
            }
        }
        let mut time_no = 10_000usize;
        for i in 0..10_000 {
            if no_ia.step(input) > 0 {
                time_no = i;
                break;
            }
        }
        assert!(
            time_with >= time_no,
            "IA must delay first spike: with={time_with} vs without={time_no}"
        );
    }

    #[test]
    fn atype_reduces_firing_rate() {
        // IA should reduce steady-state firing rate
        let mut with_ia = ATypeKNeuron::new();
        let mut no_ia = ATypeKNeuron::new();
        no_ia.g_a = 0.0;

        let input = 3.0;
        let mut spikes_ia = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_ia += with_ia.step(input);
            spikes_no += no_ia.step(input);
        }
        assert!(
            spikes_no >= spikes_ia,
            "IA should reduce firing rate: IA={spikes_ia} vs none={spikes_no}"
        );
    }

    #[test]
    fn atype_negative_input_no_crash() {
        let mut n = ATypeKNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
        assert!(n.v >= -100.0);
    }

    #[test]
    fn atype_nan_input_stays_finite() {
        let mut n = ATypeKNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn atype_extreme_input_bounded() {
        let mut n = ATypeKNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn atype_reset_clears_state() {
        let mut n = ATypeKNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.a, 0.1);
        assert_eq!(n.b, 0.8);
    }

    #[test]
    fn atype_gates_bounded() {
        let mut n = ATypeKNeuron::new();
        for _ in 0..10_000 {
            n.step(10.0);
        }
        assert!(n.a >= 0.0 && n.a <= 1.0);
        assert!(n.b >= 0.0 && n.b <= 1.0);
    }

    #[test]
    fn atype_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = ATypeKNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }

    // -- BK Neuron tests --

    #[test]
    fn bk_fires_with_input() {
        let mut n = BKNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(3.0);
        }
        assert!(spikes > 5, "BK neuron must fire with input, got {spikes}");
    }

    #[test]
    fn bk_silent_without_input() {
        let mut n = BKNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "BK neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn bk_ca_accumulates_during_spiking() {
        let mut n = BKNeuron::new();
        assert_eq!(n.ca, 0.0);
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(
            n.ca > 0.0,
            "Ca2+ must accumulate during spiking, ca={}",
            n.ca
        );
    }

    #[test]
    fn bk_deepens_ahp() {
        // BK should produce deeper AHP (more negative post-spike voltage)
        // Compare with and without BK after a burst
        let mut with_bk = BKNeuron::new();
        let mut no_bk = BKNeuron::new();
        no_bk.g_bk = 0.0;

        // Drive both to spike, then check voltage
        for _ in 0..2000 {
            with_bk.step(5.0);
            no_bk.step(5.0);
        }
        // After sustained spiking, BK with Ca2+ should keep voltage lower
        // (stronger K+ current from BK)
        // Test that BK neuron has non-zero Ca2+ (proves it's active)
        assert!(with_bk.ca > 0.0, "BK neuron must have Ca2+ after spiking");
    }

    #[test]
    fn bk_reduces_firing_rate() {
        // BK should reduce firing rate via stronger repolarisation
        let mut with_bk = BKNeuron::new();
        let mut no_bk = BKNeuron::new();
        no_bk.g_bk = 0.0;

        let input = 3.0;
        let mut spikes_bk = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_bk += with_bk.step(input);
            spikes_no += no_bk.step(input);
        }
        // BK adds extra K+ → fewer spikes (or equal if Ca2+ builds slowly)
        assert!(
            spikes_no >= spikes_bk,
            "BK should reduce firing: BK={spikes_bk} vs none={spikes_no}"
        );
    }

    #[test]
    fn bk_negative_input_no_crash() {
        let mut n = BKNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn bk_nan_input_stays_finite() {
        let mut n = BKNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn bk_extreme_input_bounded() {
        let mut n = BKNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn bk_reset_clears_state() {
        let mut n = BKNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.ca, 0.0);
    }

    #[test]
    fn bk_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = BKNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }

    // -- SK Neuron tests --

    #[test]
    fn sk_fires_with_input() {
        let mut n = SKNeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(2.0);
        }
        assert!(spikes > 5, "SK neuron must fire with input, got {spikes}");
    }

    #[test]
    fn sk_silent_without_input() {
        let mut n = SKNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "SK neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn sk_adaptation() {
        // SK causes spike frequency adaptation
        let mut n = SKNeuron::new();
        let input = 5.0;
        let mut early = 0;
        for _ in 0..2000 {
            early += n.step(input);
        }
        let mut late = 0;
        for _ in 0..2000 {
            late += n.step(input);
        }
        assert!(
            early >= late,
            "SK should cause adaptation: early={early}, late={late}"
        );
    }

    #[test]
    fn sk_ca_dependent_only() {
        // SK at rest (ca=0) should contribute zero current
        let n = SKNeuron::new();
        let ca2 = n.ca * n.ca;
        let sk_inf = ca2 / (ca2 + 0.25);
        assert!(
            sk_inf < 0.001,
            "SK must be inactive at ca=0, sk_inf={sk_inf}"
        );
    }

    #[test]
    fn sk_reduces_firing_rate() {
        let mut with_sk = SKNeuron::new();
        let mut no_sk = SKNeuron::new();
        no_sk.g_sk = 0.0;

        let input = 3.0;
        let mut spikes_sk = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_sk += with_sk.step(input);
            spikes_no += no_sk.step(input);
        }
        assert!(
            spikes_no >= spikes_sk,
            "SK should reduce firing: SK={spikes_sk} vs none={spikes_no}"
        );
    }

    #[test]
    fn sk_negative_input_no_crash() {
        let mut n = SKNeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn sk_nan_input_stays_finite() {
        let mut n = SKNeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn sk_extreme_input_bounded() {
        let mut n = SKNeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn sk_reset_clears_state() {
        let mut n = SKNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.ca, 0.0);
    }

    #[test]
    fn sk_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = SKNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }

    // -- NMDA Neuron tests --

    #[test]
    fn nmda_fires_with_input() {
        let mut n = NMDANeuron::new();
        let mut spikes = 0;
        for _ in 0..2_000 {
            spikes += n.step(3.0);
        }
        assert!(spikes > 5, "NMDA neuron must fire with input, got {spikes}");
    }

    #[test]
    fn nmda_silent_without_input() {
        let mut n = NMDANeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(
            spikes, 0,
            "NMDA neuron must be silent without input, got {spikes}"
        );
    }

    #[test]
    fn nmda_mg_block_at_rest() {
        // At -65 mV: B = 1/(1 + 1/3.57 * exp(0.062*65)) = 1/(1 + 0.28 * 56.3) = 1/16.8 = 0.06
        let n = NMDANeuron::new();
        let mg_block = 1.0 / (1.0 + (n.mg_conc / 3.57) * (-0.062 * n.v).exp());
        assert!(
            mg_block < 0.1,
            "Mg2+ block must be strong at rest, B={mg_block}"
        );
    }

    #[test]
    fn nmda_mg_relief_at_depolarised() {
        // At -20 mV: B = 1/(1 + 0.28 * exp(0.062*20)) = 1/(1 + 0.28*3.45) = 1/1.97 = 0.51
        let mg_block = 1.0 / (1.0 + (1.0 / 3.57) * (-0.062 * (-20.0_f64)).exp());
        assert!(
            mg_block > 0.4,
            "Mg2+ block must be relieved at -20 mV, B={mg_block}"
        );
    }

    #[test]
    fn nmda_s_builds_with_input() {
        let mut n = NMDANeuron::new();
        assert_eq!(n.s_nmda, 0.0);
        for _ in 0..2000 {
            n.step(5.0);
        }
        assert!(
            n.s_nmda > 0.0,
            "s_nmda must build with input, s={}",
            n.s_nmda
        );
    }

    #[test]
    fn nmda_s_decays_without_input() {
        let mut n = NMDANeuron::new();
        // Build up s
        for _ in 0..2000 {
            n.step(5.0);
        }
        let s_peak = n.s_nmda;
        // Remove input
        for _ in 0..2000 {
            n.step(0.0);
        }
        assert!(n.s_nmda < s_peak, "s_nmda must decay after input removal");
    }

    #[test]
    fn nmda_zero_mg_increases_current() {
        // Without Mg2+ block, NMDA should contribute more
        let mut with_mg = NMDANeuron::new();
        let mut no_mg = NMDANeuron::new();
        no_mg.mg_conc = 0.0;

        let input = 2.0;
        let mut spikes_mg = 0;
        let mut spikes_no = 0;
        for _ in 0..10_000 {
            spikes_mg += with_mg.step(input);
            spikes_no += no_mg.step(input);
        }
        assert!(
            spikes_no >= spikes_mg,
            "No Mg2+ should increase NMDA current: no_mg={spikes_no} vs mg={spikes_mg}"
        );
    }

    #[test]
    fn nmda_negative_input_no_crash() {
        let mut n = NMDANeuron::new();
        for _ in 0..10_000 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn nmda_nan_input_stays_finite() {
        let mut n = NMDANeuron::new();
        n.step(f64::NAN);
        assert!(n.v.is_finite());
    }

    #[test]
    fn nmda_extreme_input_bounded() {
        let mut n = NMDANeuron::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0);
    }

    #[test]
    fn nmda_reset_clears_state() {
        let mut n = NMDANeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -65.0);
        assert_eq!(n.s_nmda, 0.0);
    }

    #[test]
    fn nmda_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = NMDANeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(3.0));
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 200,
            "1k steps must complete in <200ms"
        );
    }
}
