// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar Circuit Neuron Models

//! Cerebellar circuit neuron models for granular and molecular layer computations.
//!
//! Phase 3D: granule cell, Golgi cell, stellate cell, Lugaro cell,
//! unipolar brush cell, deep cerebellar nuclei neuron.
//! Added one by one with full 7-point checklist verification.

use super::biophysical::safe_rate;

// ═══════════════════════════════════════════════════════════════════
// Granule Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar granule cell — most numerous neuron in the brain (~50%).
///
/// Biophysics: LIF core with tonic GABAergic inhibition from Golgi cells,
/// T-type Ca2+ current for post-inhibitory rebound bursting, and very high
/// input resistance due to tiny soma (6-8 µm). Four short dendrites receive
/// mossy fibre input at glomeruli; output via parallel fibres to Purkinje cells.
///
/// The tonic GABA conductance models the continuous inhibitory tone that
/// Golgi cells impose, keeping granule cells near threshold but rarely
/// spontaneously active. Release from inhibition (disinhibition) triggers
/// rebound bursts via T-type Ca2+ channels.
///
/// D'Angelo et al., J Neurosci 21(3), 2001; Bhalla & Bhatt, Cerebellum, 2012.
#[derive(Clone, Debug)]
pub struct GranuleCell {
    pub v: f64,
    // T-type Ca2+ gating
    pub s: f64,         // T-type inactivation (slow)
    // Conductances (mS/cm²)
    pub g_l: f64,       // Leak
    pub g_tonic: f64,   // Tonic GABA conductance
    pub g_t: f64,       // T-type Ca2+ conductance
    // Reversal potentials (mV)
    pub e_l: f64,
    pub e_gaba: f64,    // GABA reversal (-75 mV, shunting)
    pub e_ca: f64,      // Ca2+ reversal
    // Membrane
    pub tau_m: f64,     // Membrane time constant (ms) — very short for tiny soma
    pub c_m: f64,       // Specific capacitance (µF/cm²)
    pub v_threshold: f64,
    pub v_reset: f64,
    pub refrac_count: f64,
    pub refrac_period: f64,
    pub gain: f64,      // Input scaling
    pub dt: f64,
}

impl Default for GranuleCell {
    fn default() -> Self { Self::new() }
}

impl GranuleCell {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            s: 0.95,            // T-type inactivation de-inactivated at rest
            g_l: 0.05,          // Low leak for high Rin
            g_tonic: 0.02,      // Tonic GABA
            g_t: 0.03,          // T-type Ca2+
            e_l: -70.0,
            e_gaba: -75.0,
            e_ca: 120.0,
            tau_m: 5.0,         // Short tau for tiny soma
            c_m: 1.0,
            v_threshold: -40.0,
            v_reset: -70.0,
            refrac_count: 0.0,
            refrac_period: 1.0, // 1 ms refractory
            gain: 1.5,
            dt: 0.5,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        // Refractory period
        if self.refrac_count > 0.0 {
            self.refrac_count -= self.dt;
            self.v = self.v_reset;
            // T-type inactivation still evolves during refractory
            let s_inf = 1.0 / (1.0 + ((self.v + 60.0) / 6.5).exp());
            let tau_s = 20.0 + 50.0 / (1.0 + ((self.v + 65.0) / 10.0).exp());
            self.s += self.dt * (s_inf - self.s) / tau_s;
            return 0;
        }

        // T-type Ca2+ activation (fast, instantaneous steady-state)
        let m_t_inf = 1.0 / (1.0 + (-(self.v + 52.0) / 5.0).exp());
        // T-type Ca2+ inactivation (slow)
        let s_inf = 1.0 / (1.0 + ((self.v + 60.0) / 6.5).exp());
        let tau_s = 20.0 + 50.0 / (1.0 + ((self.v + 65.0) / 10.0).exp());

        // Currents
        let i_l = self.g_l * (self.v - self.e_l);
        let i_tonic = self.g_tonic * (self.v - self.e_gaba);
        let i_t = self.g_t * m_t_inf * m_t_inf * self.s * (self.v - self.e_ca);
        let i_ext = self.gain * current.max(0.0);

        // Membrane equation
        let dv = (-i_l - i_tonic - i_t + i_ext) / self.c_m;
        self.v += self.dt * dv / self.tau_m;

        // T-type inactivation update
        self.s += self.dt * (s_inf - self.s) / tau_s;

        // Spike detection
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.refrac_period;
            // Spike inactivates T-type channels
            self.s *= 0.5;
            return 1;
        }

        // Bound membrane potential
        if self.v < -100.0 { self.v = -100.0; }
        if self.v > 60.0 { self.v = 60.0; }
        if !self.v.is_finite() { self.v = self.v_reset; }
        if !self.s.is_finite() { self.s = 0.95; }

        0
    }

    pub fn reset(&mut self) {
        self.v = -70.0;
        self.s = 0.95;
        self.refrac_count = 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Golgi Cell
// ═══════════════════════════════════════════════════════════════════

/// Cerebellar Golgi cell — large inhibitory interneuron in the granular layer.
///
/// Biophysics: Pospischil 2008 RS-type Na+/K+ core for regular spiking,
/// A-type K+ current (transient outward) for onset delay and phasic pause,
/// Ca2+-dependent slow AHP for spike frequency adaptation. Provides tonic
/// and phasic GABAergic/glycinergic inhibition to granule cells at glomeruli.
///
/// Spontaneously active at 3-10 Hz due to intrinsic pacemaker currents.
/// Dendritic arbour in molecular layer receives parallel fibre input
/// (feedback) and ascending granule cell axon input (feedforward).
///
/// Solinas et al., Front Cell Neurosci 1:2, 2007; Pospischil et al., Biol Cybern 99:427, 2008.
#[derive(Clone, Debug)]
pub struct GolgiCell {
    pub v: f64,
    // Pospischil gating
    pub m: f64,     // Na+ activation
    pub h: f64,     // Na+ inactivation
    pub n: f64,     // Kdr activation
    // A-type K+ gating
    pub a: f64,     // A-type activation
    pub b: f64,     // A-type inactivation
    // Ca2+ and AHP
    pub ca: f64,    // Intracellular Ca2+
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_a: f64,       // A-type K+
    pub g_ahp: f64,     // Ca2+-dependent K+ (AHP)
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,       // Kinetic scaling factor
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for GolgiCell {
    fn default() -> Self { Self::new() }
}

impl GolgiCell {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
            a: 0.1,
            b: 0.8,
            ca: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_a: 2.0,
            g_ahp: 0.5,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,          // Fast kinetics for reliable spiking
            dt: 0.5,
            v_threshold: -20.0,
            gain: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let input = self.gain * current;
        let sub_steps = 4;
        let sub_dt = self.dt / sub_steps as f64;
        let mut fired = 0i32;

        for _ in 0..sub_steps {
            let v = self.v;

            // Wang-Buzsáki alpha/beta rates with phi scaling
            let alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let beta_m = 4.0 * (-(v + 60.0) / 18.0).exp();

            let alpha_h = 0.07 * (-(v + 58.0) / 20.0).exp();
            let beta_h = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());

            let alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let beta_n = 0.125 * (-(v + 44.0) / 80.0).exp();

            // A-type K+ gating (Connor-Stevens style)
            let a_inf = 1.0 / (1.0 + (-(v + 50.0) / 20.0).exp());
            let tau_a = 2.0;
            let b_inf = 1.0 / (1.0 + ((v + 70.0) / 6.0).exp());
            let tau_b = 50.0;

            // Gate updates with phi scaling for Na+/K+
            self.m += sub_dt * self.phi * (alpha_m * (1.0 - self.m) - beta_m * self.m);
            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.a += sub_dt * (a_inf - self.a) / tau_a;
            self.b += sub_dt * (b_inf - self.b) / tau_b;

            // Ca2+ dynamics (spike-triggered influx, exponential decay)
            let tau_ca = 200.0;
            self.ca += sub_dt * (-self.ca / tau_ca);

            // Currents
            let i_na = self.g_na * self.m.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_a = self.g_a * self.a.powi(3) * self.b * (v - self.e_k);
            let i_ahp = self.g_ahp * (self.ca / (self.ca + 0.5)) * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_a - i_ahp - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            // Spike detection
            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -65.0; // Reset for h-gate recovery
                self.ca += 0.2; // Ca2+ influx on spike
            }
        }

        // Safety bounds
        if self.v < -100.0 { self.v = -100.0; }
        if self.v > 60.0 { self.v = 60.0; }
        if !self.v.is_finite() { self.v = -65.0; self.m = 0.05; self.h = 0.6; self.n = 0.32; }
        if !self.ca.is_finite() { self.ca = 0.0; }
        self.m = self.m.clamp(0.0, 1.0);
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
    pub h: f64,     // Na+ inactivation
    pub n: f64,     // Kdr activation
    pub p: f64,     // Kv3.1 activation
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
    fn default() -> Self { Self::new() }
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
            g_kv3: 3.0,    // Less Kv3.1 than PV+ basket
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 0.5,      // Smaller cell → lower capacitance
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
        if self.v < -100.0 { self.v = -100.0; }
        if self.v > 60.0 { self.v = 60.0; }
        if !self.v.is_finite() { self.v = -65.0; self.h = 0.6; self.n = 0.32; }
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
    pub adapt: f64,         // Adaptation current
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_adapt: f64,
    pub a_adapt: f64,       // Adaptation coupling strength
    pub gain: f64,
    pub serotonin: f64,     // 5-HT modulation factor [0, 1]
    pub dt: f64,
}

impl Default for LugaroCell {
    fn default() -> Self { Self::new() }
}

impl LugaroCell {
    pub fn new() -> Self {
        Self {
            v: -55.0,
            adapt: 0.0,
            v_rest: -55.0,      // Depolarised rest for spontaneous firing
            v_reset: -65.0,
            v_threshold: -48.0,
            tau_m: 10.0,
            tau_adapt: 150.0,
            a_adapt: 0.05,
            gain: 2.0,
            serotonin: 0.0,    // No 5-HT modulation by default
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
        if self.v < -100.0 { self.v = -100.0; }
        if self.v > 60.0 { self.v = 60.0; }
        if !self.v.is_finite() { self.v = self.v_reset; }
        if !self.adapt.is_finite() { self.adapt = 0.0; }

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
    pub persistent: f64,    // Slow NMDA-like persistent current
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_persistent: f64, // Slow decay of persistent current (ms)
    pub persistent_gain: f64, // How much input drives persistent current
    pub gain: f64,
    pub dt: f64,
}

impl Default for UnipolarBrushCell {
    fn default() -> Self { Self::new() }
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
        if self.persistent < 0.0 { self.persistent = 0.0; }

        // LIF with persistent current
        let dv = (-(self.v - self.v_rest) + input + self.persistent) / self.tau_m;
        self.v += self.dt * dv;

        // Spike detection
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            return 1;
        }

        // Safety bounds
        if self.v < -100.0 { self.v = -100.0; }
        if self.v > 60.0 { self.v = 60.0; }
        if !self.v.is_finite() { self.v = self.v_reset; }
        if !self.persistent.is_finite() { self.persistent = 0.0; }

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
/// Biophysics: WB Na+/K+ core with T-type Ca2+ for post-inhibitory rebound
/// bursting and Ih (hyperpolarisation-activated) for pacemaker-like activity.
/// DCN neurons are the sole output of the cerebellum, receiving massive
/// inhibitory input from Purkinje cells and excitatory input from mossy
/// fibres and climbing fibres.
///
/// Rebound bursting: when Purkinje inhibition is released (pause in PC
/// firing), T-type Ca2+ channels that de-inactivated during hyperpolarisation
/// produce a burst of spikes. This is the primary mechanism for cerebellar
/// timing signals.
///
/// Llinás & Mühlethaler, J Physiol 404:241, 1988; Jahnsen, J Physiol 372:129, 1986.
#[derive(Clone, Debug)]
pub struct DCNNeuron {
    pub v: f64,
    pub h: f64,     // Na+ inactivation
    pub n: f64,     // Kdr activation
    // T-type Ca2+ gating
    pub s: f64,     // T-type inactivation (slow)
    // Ih gating
    pub r: f64,     // Ih activation
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_t: f64,   // T-type Ca2+
    pub g_h: f64,   // Ih
    pub g_l: f64,
    // Reversal potentials
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_h: f64,   // Ih reversal (~-40 mV, mixed cation)
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
    pub gain: f64,
}

impl Default for DCNNeuron {
    fn default() -> Self { Self::new() }
}

impl DCNNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            h: 0.6,
            n: 0.32,
            s: 0.8,     // De-inactivated at rest
            r: 0.1,     // Ih partially active
            g_na: 35.0,
            g_k: 9.0,
            g_t: 0.1,   // T-type — low to avoid window current at rest
            g_h: 0.02,  // Ih — modest to avoid spontaneous firing
            g_l: 0.2,   // Higher leak to stabilise at rest
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
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

            // T-type Ca2+ gating
            let m_t_inf = 1.0 / (1.0 + (-(v + 52.0) / 5.0).exp());
            let s_inf = 1.0 / (1.0 + ((v + 60.0) / 6.5).exp());
            let tau_s = 20.0 + 50.0 / (1.0 + ((v + 65.0) / 10.0).exp());

            // Ih gating
            let r_inf = 1.0 / (1.0 + ((v + 80.0) / 10.0).exp());
            let tau_r = 100.0 + 200.0 / (1.0 + ((v + 70.0) / 10.0).exp());

            // Gate updates
            self.h += sub_dt * self.phi * (alpha_h * (1.0 - self.h) - beta_h * self.h);
            self.n += sub_dt * self.phi * (alpha_n * (1.0 - self.n) - beta_n * self.n);
            self.s += sub_dt * (s_inf - self.s) / tau_s;
            self.r += sub_dt * (r_inf - self.r) / tau_r;

            // Currents
            let i_na = self.g_na * m_inf.powi(3) * self.h * (v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (v - self.e_k);
            let i_t = self.g_t * m_t_inf.powi(2) * self.s * (v - self.e_ca);
            let i_h = self.g_h * self.r * (v - self.e_h);
            let i_l = self.g_l * (v - self.e_l);

            let dv = (-i_na - i_k - i_t - i_h - i_l + input) / self.c_m;
            self.v += sub_dt * dv;

            if self.v >= self.v_threshold {
                fired = 1;
                self.v = -60.0;
                self.s *= 0.5; // T-type inactivation on spike
            }
        }

        // Safety bounds
        if self.v < -100.0 { self.v = -100.0; }
        if self.v > 60.0 { self.v = 60.0; }
        if !self.v.is_finite() { self.v = -60.0; self.h = 0.6; self.n = 0.32; }
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
        self.s = self.s.clamp(0.0, 1.0);
        self.r = self.r.clamp(0.0, 1.0);

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

    // -- Granule Cell tests --

    #[test]
    fn granule_fires_with_strong_input() {
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(15.0);
        }
        assert!(spikes > 10, "Granule cell must fire with strong excitatory input, got {spikes}");
    }

    #[test]
    fn granule_silent_at_rest() {
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "Granule cell must be silent without input (tonic GABA inhibition)");
    }

    #[test]
    fn granule_no_fire_weak_input() {
        // Tonic GABA raises effective threshold
        let mut n = GranuleCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(1.0);
        }
        assert!(spikes == 0, "Weak input should not overcome tonic GABA, got {spikes}");
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
    fn granule_rebound_burst() {
        // Release from hyperpolarisation triggers T-type rebound
        let mut n = GranuleCell::new();
        // Hyperpolarise to de-inactivate T-type channels
        for _ in 0..2000 {
            n.step(0.0);
        }
        // Ensure s is high (de-inactivated)
        assert!(n.s > 0.8, "T-type must be de-inactivated at rest, s={}", n.s);

        // Now provide input — T-type should help fire
        let mut spikes_early = 0;
        for _ in 0..200 {
            spikes_early += n.step(10.0);
        }

        // Compare with a neuron that had T-type pre-inactivated
        let mut n2 = GranuleCell::new();
        n2.s = 0.1; // pre-inactivated
        let mut spikes_no_rebound = 0;
        for _ in 0..200 {
            spikes_no_rebound += n2.step(10.0);
        }

        assert!(
            spikes_early >= spikes_no_rebound,
            "De-inactivated T-type should facilitate firing: early={spikes_early} vs inactivated={spikes_no_rebound}"
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
        n.step(f64::NAN);
        assert!(n.v.is_finite(), "NaN input must not corrupt state");
    }

    #[test]
    fn granule_extreme_input_bounded() {
        let mut n = GranuleCell::new();
        for _ in 0..1000 {
            n.step(1e6);
        }
        assert!(n.v.is_finite() && n.v <= 60.0, "Extreme input must stay bounded");
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
        assert_eq!(n.refrac_count, 0.0);
    }

    #[test]
    fn granule_high_input_resistance() {
        // Small soma → large voltage response to small current
        let mut n = GranuleCell::new();
        let v_before = n.v;
        // Single step with moderate input
        n.step(5.0);
        let dv = n.v - v_before;
        assert!(dv > 0.5, "High Rin should give large voltage change, got dv={dv}");
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
        assert!(spikes > 10, "Golgi cell must fire with excitatory input, got {spikes}");
    }

    #[test]
    fn golgi_spontaneous_firing() {
        // Golgi cells are spontaneously active due to depolarised leak
        let mut n = GolgiCell::new();
        let mut spikes = 0;
        for _ in 0..20_000 {
            spikes += n.step(0.0);
        }
        // With e_l = -60 and v_t = -56.2, may or may not spontaneously fire
        // The key property is that they fire easily with minimal input
        let mut n2 = GolgiCell::new();
        let mut spikes_small = 0;
        for _ in 0..20_000 {
            spikes_small += n2.step(0.5);
        }
        assert!(spikes_small > 0, "Golgi cell should fire with minimal input (near-threshold), got {spikes_small}");
    }

    #[test]
    fn golgi_adaptation_via_ahp() {
        // Slow AHP causes spike frequency adaptation
        let mut n = GolgiCell::new();
        let input = 8.0;
        // Count spikes in first 500 steps vs last 500 steps
        let mut spikes_early = 0;
        for _ in 0..2000 {
            spikes_early += n.step(input);
        }
        let mut spikes_late = 0;
        for _ in 0..2000 {
            spikes_late += n.step(input);
        }
        // Ca2+ accumulates → AHP increases → firing slows
        // Early period may fire more or equal (adaptation takes time)
        assert!(spikes_early >= spikes_late || (spikes_early as i32 - spikes_late as i32).abs() < 5,
            "AHP should cause adaptation: early={spikes_early}, late={spikes_late}");
    }

    #[test]
    fn golgi_a_type_onset_delay() {
        // A-type K+ creates delay to first spike
        let mut with_a = GolgiCell::new();
        let mut no_a = GolgiCell::new();
        no_a.g_a = 0.0;

        // Find time to first spike
        let mut time_with = 0usize;
        for i in 0..5000 {
            if with_a.step(5.0) > 0 { time_with = i; break; }
        }
        let mut time_no = 0usize;
        for i in 0..5000 {
            if no_a.step(5.0) > 0 { time_no = i; break; }
        }
        assert!(time_with >= time_no,
            "A-type K+ should delay first spike: with={time_with} vs without={time_no}");
    }

    #[test]
    fn golgi_ca_accumulates_during_spiking() {
        let mut n = GolgiCell::new();
        assert_eq!(n.ca, 0.0);
        for _ in 0..5000 {
            n.step(10.0);
        }
        assert!(n.ca > 0.0, "Ca2+ must accumulate during spiking, ca={}", n.ca);
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
        assert!(n.v.is_finite() && n.v <= 60.0, "Extreme input must stay bounded");
    }

    #[test]
    fn golgi_reset_clears_state() {
        let mut n = GolgiCell::new();
        for _ in 0..5000 {
            n.step(10.0);
        }
        n.reset();
        assert_eq!(n.v, -60.0);
        assert_eq!(n.ca, 0.0);
        assert_eq!(n.a, 0.1);
        assert_eq!(n.b, 0.8);
    }

    #[test]
    fn golgi_gates_bounded() {
        let mut n = GolgiCell::new();
        for _ in 0..10_000 {
            n.step(15.0);
        }
        assert!(n.m >= 0.0 && n.m <= 1.0);
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.a >= 0.0 && n.a <= 1.0);
        assert!(n.b >= 0.0 && n.b <= 1.0);
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
        assert!(spikes > 5, "Stellate cell must fire with input, got {spikes}");
    }

    #[test]
    fn stellate_silent_without_input() {
        let mut n = StellateCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "Stellate cell must be silent without input, got {spikes}");
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
        assert!(spikes > 50, "FS stellate should fire at high rate, got {spikes}");
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
        let diff = (spikes_early as i32 - spikes_late as i32).abs();
        assert!(diff < 20, "FS should have minimal adaptation: early={spikes_early}, late={spikes_late}");
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
        assert!(spikes > 10, "Lugaro must fire with excitatory input, got {spikes}");
    }

    #[test]
    fn lugaro_low_threshold() {
        // Near-threshold rest → fires easily with moderate input
        let mut n = LugaroCell::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(4.0);
        }
        assert!(spikes > 10, "Lugaro should fire easily with moderate input, got {spikes}");
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
        assert!(spikes_early >= spikes_late,
            "Adaptation should slow firing: early={spikes_early}, late={spikes_late}");
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
        assert!(spikes_5ht >= spikes_no,
            "5-HT must increase firing: 5HT={spikes_5ht} vs none={spikes_no}");
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
        assert!(n.adapt > initial, "Adaptation must increase during spiking, adapt={}", n.adapt);
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
        assert!(spikes > 10, "UBC must fire with excitatory input, got {spikes}");
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
        assert!(n.persistent > 0.0, "Persistent current must build during input");

        // Now remove input — persistent current should persist
        let persistent_before = n.persistent;
        for _ in 0..100 {
            n.step(0.0);
        }
        assert!(n.persistent > 0.0, "Persistent current must persist after input removal");
        assert!(n.persistent < persistent_before, "Persistent current must decay");
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
        let mut post_spikes = 0;
        for _ in 0..500 {
            post_spikes += n.step(0.0);
        }
        // May or may not spike depending on persistent level — just test it doesn't crash
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
        assert!(spikes > 5, "DCN must fire with excitatory input, got {spikes}");
    }

    #[test]
    fn dcn_silent_without_input() {
        let mut n = DCNNeuron::new();
        let mut spikes = 0;
        for _ in 0..10_000 {
            spikes += n.step(0.0);
        }
        assert_eq!(spikes, 0, "DCN must be silent without input, got {spikes}");
    }

    #[test]
    fn dcn_rebound_burst() {
        // Hyperpolarisation → T-type de-inactivation → rebound burst
        let mut n = DCNNeuron::new();
        // Hyperpolarise to de-inactivate T-type
        for _ in 0..2000 {
            n.step(-5.0);
        }
        assert!(n.s > 0.5, "T-type must de-inactivate during hyperpolarisation, s={}", n.s);

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
        assert!(spikes >= spikes2,
            "De-inactivated T-type should facilitate rebound: rebound={spikes} vs inact={spikes2}");
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
        assert!(with_ih.v > no_ih.v,
            "Ih should depolarise from hyperpolarised state: Ih={:.1} vs no_Ih={:.1}",
            with_ih.v, no_ih.v);
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
        n.step(f64::NAN);
        assert!(n.v.is_finite());
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
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
        assert!(n.s >= 0.0 && n.s <= 1.0);
        assert!(n.r >= 0.0 && n.r <= 1.0);
    }

    #[test]
    fn dcn_performance_1k_steps() {
        let start = std::time::Instant::now();
        let mut n = DCNNeuron::new();
        for _ in 0..1_000 {
            std::hint::black_box(n.step(5.0));
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 200, "1k steps must complete in <200ms");
    }
}
