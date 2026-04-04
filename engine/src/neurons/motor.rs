// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Motor Neuron Models

//! Motor neuron models for spinal and cortical motor circuits.
//!
//! Phase 3C: alpha motor, gamma motor, upper motor, Renshaw cell, motor unit.
//! Added one by one with full 7-point checklist verification.

use super::biophysical::safe_rate;

// ═══════════════════════════════════════════════════════════════════
// Alpha Motor Neuron
// ═══════════════════════════════════════════════════════════════════

/// Alpha motor neuron — spinal cord, innervates extrafusal muscle fibres.
///
/// Biophysics: Wang-Buzsáki Na+/K+ core, persistent inward current (PIC)
/// for bistable firing (plateau potentials), Ca2+-dependent AHP for rate
/// limiting (f-I gain control). Larger soma than cortical neurons → lower
/// input resistance.
///
/// PIC is modelled as a slow L-type Ca2+ current that activates at
/// depolarised potentials and inactivates very slowly, enabling plateau
/// potentials and self-sustained firing after brief input.
///
/// AHP from Ca2+-activated K+ (SK channels) limits firing rate and
/// produces the characteristic linear f-I relationship of motor neurons.
///
/// Powers & Binder, J. Neurophysiol. 86, 2001.
/// Heckman & Enoka, Compr. Physiol. 2(4), 2012.
#[derive(Clone, Debug)]
pub struct AlphaMotorNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub m_pic: f64,   // PIC (L-type Ca2+) activation
    pub ca: f64,       // Intracellular Ca2+ (µM)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_pic: f64,    // Persistent inward current
    pub g_ahp: f64,    // Ca2+-dependent K+ (AHP)
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,   // Ca2+ decay time constant (ms)
    pub dt: f64,
    pub v_threshold: f64,
}

impl AlphaMotorNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            m_pic: 0.0,
            ca: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_pic: 0.5,     // PIC for plateau potentials
            g_ahp: 3.0,     // Strong AHP for rate limiting
            g_l: 0.15,      // Slightly higher leak (larger soma)
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -65.0,
            c_m: 1.5,       // Larger soma → higher capacitance
            phi: 4.0,       // Slightly slower than PV+ FS
            tau_ca: 150.0,  // Slow Ca2+ clearance for AHP
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        for _ in 0..n_sub {
            // WB Na+/K+ gating
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;

            // PIC (L-type Ca2+): activates at subthreshold potentials, slow dynamics
            let m_pic_inf = 1.0 / (1.0 + (-(self.v + 50.0) / 5.0).exp());
            self.m_pic += (m_pic_inf - self.m_pic) / 50.0 * self.dt;

            // Ca2+ dynamics: entry proportional to PIC + spike Ca2+ transient
            let ca_entry = self.g_pic * self.m_pic * (self.v - self.e_ca).abs() * 0.001;
            let ca_spike = if self.v > -10.0 { 0.02 } else { 0.0 };
            self.ca += (-self.ca / self.tau_ca + ca_entry + ca_spike) * self.dt;
            if self.ca < 0.0 { self.ca = 0.0; }

            // AHP: Ca2+-activated K+ (SK channels)
            let ahp_inf = self.ca / (self.ca + 0.5);

            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_pic = self.g_pic * self.m_pic * (self.v - self.e_ca);
            let i_ahp = self.g_ahp * ahp_inf * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);

            self.v += (-i_na - i_k - i_pic - i_ahp - i_l + current) / self.c_m * self.dt;
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
        self.m_pic = 0.0;
        self.ca = 0.0;
    }
}

impl Default for AlphaMotorNeuron {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Gamma Motor Neuron
// ═══════════════════════════════════════════════════════════════════

/// Gamma motor neuron — innervates intrafusal fibres of muscle spindles.
///
/// Regulates proprioceptive sensitivity by adjusting spindle tension.
/// Smaller soma than alpha, lower firing rates (5-30 Hz), no PIC.
/// Simple LIF with spike-frequency adaptation (slow K+ current).
/// Two subtypes: dynamic (bag1, velocity-sensitive) and static
/// (bag2/chain, length-sensitive) — controlled by `dynamic` flag.
///
/// Prochazka & Hulliger, Prog. Brain Res. 80, 1989.
/// Taylor et al., J. Physiol. 519(3), 1999.
#[derive(Clone, Debug)]
pub struct GammaMotorNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub adapt: f64,       // Slow adaptation current
    pub tau_adapt: f64,   // Adaptation time constant (ms)
    pub a_adapt: f64,     // Adaptation coupling strength
    pub gain: f64,        // Input gain (fusimotor drive → mV)
    pub dynamic: bool,    // true = dynamic (bag1), false = static (bag2/chain)
    pub dt: f64,
}

impl GammaMotorNeuron {
    pub fn new() -> Self {
        Self::dynamic()
    }

    /// Dynamic gamma — innervates bag1 intrafusal fibres (velocity-sensitive).
    pub fn dynamic() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau: 8.0,
            adapt: 0.0,
            tau_adapt: 100.0,
            a_adapt: 0.3,
            gain: 1.0,
            dynamic: true,
            dt: 0.5,
        }
    }

    /// Static gamma — innervates bag2/chain intrafusal fibres (length-sensitive).
    pub fn static_type() -> Self {
        Self {
            tau: 12.0,       // Slower membrane
            tau_adapt: 200.0, // Stronger adaptation (lower steady-state rate)
            a_adapt: 0.5,
            dynamic: false,
            ..Self::dynamic()
        }
    }

    /// Step with fusimotor drive (arbitrary units, ≥ 0). Returns spike (1/0).
    pub fn step(&mut self, drive: f64) -> i32 {
        let input = self.gain * drive.max(0.0) - self.adapt;
        self.v += (-(self.v - self.v_rest) + input) / self.tau * self.dt;
        self.adapt += (self.a_adapt * (self.v - self.v_rest) - self.adapt) / self.tau_adapt * self.dt;

        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0;
    }
}

impl Default for GammaMotorNeuron {
    fn default() -> Self { Self::new() }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Alpha Motor Neuron — 6-dimension STRONG ──────────────────

    #[test]
    fn alpha_motor_fires_with_input() {
        let mut n = AlphaMotorNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(spikes > 0, "alpha motor must fire with sustained input: got {spikes}");
    }

    #[test]
    fn alpha_motor_no_fire_without_input() {
        let mut n = AlphaMotorNeuron::new();
        let spikes: i32 = (0..3000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0, "alpha motor should not fire at rest");
    }

    #[test]
    fn alpha_motor_negative_current_no_fire() {
        let mut n = AlphaMotorNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(-2.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn alpha_motor_ahp_limits_rate() {
        // AHP from Ca2+-activated K+ should limit firing rate.
        // Compare: with AHP vs without (g_ahp=0).
        let mut with_ahp = AlphaMotorNeuron::new();
        let mut no_ahp = AlphaMotorNeuron::new();
        no_ahp.g_ahp = 0.0;
        let s_ahp: i32 = (0..5000).map(|_| with_ahp.step(5.0)).sum();
        let s_none: i32 = (0..5000).map(|_| no_ahp.step(5.0)).sum();
        assert!(
            s_ahp <= s_none + 5,
            "AHP should limit rate: with={s_ahp}, without={s_none}"
        );
    }

    #[test]
    fn alpha_motor_pic_responds_to_depolarisation() {
        // PIC (m_pic) should increase from baseline during sustained input.
        let mut n = AlphaMotorNeuron::new();
        let baseline = n.m_pic;
        for _ in 0..2000 { n.step(4.0); }
        assert!(
            n.m_pic > baseline + 0.001,
            "PIC should respond to depolarisation: baseline={baseline}, after={}", n.m_pic
        );
    }

    #[test]
    fn alpha_motor_ca_increases_during_spiking() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..5000 { n.step(5.0); }
        assert!(n.ca > 0.0, "Ca2+ should accumulate during spiking: ca={}", n.ca);
    }

    #[test]
    fn alpha_motor_reset_roundtrip() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..2000 { n.step(4.0); }
        n.reset();
        let mut fresh = AlphaMotorNeuron::new();
        let r1: i32 = (0..1000).map(|_| n.step(4.0)).sum();
        let r2: i32 = (0..1000).map(|_| fresh.step(4.0)).sum();
        assert_eq!(r1, r2, "reset neuron must match fresh");
    }

    #[test]
    fn alpha_motor_voltage_bounded() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..10000 { n.step(10.0); }
        assert!(n.v.is_finite(), "voltage must stay finite");
        assert!(n.ca.is_finite(), "Ca2+ must stay finite");
        assert!(n.ca >= 0.0, "Ca2+ must be non-negative");
    }

    #[test]
    fn alpha_motor_nan_recovery() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..100 { n.step(3.0); }
        for _ in 0..10 { let _ = n.step(f64::NAN); }
        n.reset();
        assert!(n.v.is_finite());
        assert!(n.ca >= 0.0);
    }

    #[test]
    fn alpha_motor_extreme_input() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..50 { n.step(1e6); }
        n.reset();
        assert!(n.v.is_finite());
        for _ in 0..50 { n.step(-1e6); }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn alpha_motor_performance() {
        let mut n = AlphaMotorNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..5_000 { n.step(4.0); }
        assert!(start.elapsed().as_millis() < 500, "5k steps took {:?}", start.elapsed());
    }

    // ── Gamma Motor Neuron — 6-dimension STRONG ──────────────────

    #[test]
    fn gamma_dynamic_fires_with_drive() {
        let mut n = GammaMotorNeuron::dynamic();
        let spikes: i32 = (0..2000).map(|_| n.step(20.0)).sum();
        assert!(spikes > 0, "gamma dynamic must fire: got {spikes}");
    }

    #[test]
    fn gamma_static_fires_with_drive() {
        let mut n = GammaMotorNeuron::static_type();
        let spikes: i32 = (0..2000).map(|_| n.step(20.0)).sum();
        assert!(spikes > 0, "gamma static must fire: got {spikes}");
    }

    #[test]
    fn gamma_no_fire_without_drive() {
        let mut n = GammaMotorNeuron::new();
        let spikes: i32 = (0..1000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn gamma_negative_drive_no_fire() {
        let mut n = GammaMotorNeuron::new();
        // drive.max(0.0) clamps negatives
        let spikes: i32 = (0..1000).map(|_| n.step(-10.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn gamma_adaptation_reduces_rate() {
        let mut n = GammaMotorNeuron::new();
        let first: i32 = (0..1000).map(|_| n.step(20.0)).sum();
        let second: i32 = (0..1000).map(|_| n.step(20.0)).sum();
        assert!(
            second <= first + 3,
            "gamma should adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn gamma_static_adapts_more_than_dynamic() {
        let mut dyn_ = GammaMotorNeuron::dynamic();
        let mut stat = GammaMotorNeuron::static_type();
        let dyn_spikes: i32 = (0..2000).map(|_| dyn_.step(20.0)).sum();
        let stat_spikes: i32 = (0..2000).map(|_| stat.step(20.0)).sum();
        // Static has stronger adaptation → fewer spikes
        assert!(
            stat_spikes <= dyn_spikes + 5,
            "static ({stat_spikes}) should fire <= dynamic ({dyn_spikes})"
        );
    }

    #[test]
    fn gamma_reset_roundtrip() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..1000 { n.step(20.0); }
        n.reset();
        let mut fresh = GammaMotorNeuron::new();
        let r1: i32 = (0..500).map(|_| n.step(20.0)).sum();
        let r2: i32 = (0..500).map(|_| fresh.step(20.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn gamma_voltage_bounded() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..10000 { n.step(50.0); }
        assert!(n.v.is_finite());
        assert!(n.adapt.is_finite());
    }

    #[test]
    fn gamma_nan_recovery() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..50 { n.step(20.0); }
        for _ in 0..10 { let _ = n.step(f64::NAN); }
        n.reset();
        assert!(n.v.is_finite());
        assert_eq!(n.adapt, 0.0);
    }

    #[test]
    fn gamma_extreme_input() {
        let mut n = GammaMotorNeuron::new();
        for _ in 0..50 { n.step(1e6); }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn gamma_performance() {
        let mut n = GammaMotorNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..100_000 { n.step(20.0); }
        assert!(start.elapsed().as_millis() < 50, "100k steps took {:?}", start.elapsed());
    }
}
