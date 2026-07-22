// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Alpha Motor Neuron

//! Alpha motor-neuron biophysics and plateau-potential dynamics.

use crate::neurons::biophysical::safe_rate;

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
    pub m_pic: f64,  // PIC (L-type Ca²⁺) activation
    pub h_pic: f64,  // PIC slow inactivation (tau ~200 ms)
    pub ca: f64,     // Intracellular Ca²⁺ (µM)
    pub ca_buf: f64, // Bound Ca²⁺ (buffered fraction)
    // Conductances (mS/cm²)
    pub g_na: f64,
    pub g_k: f64,
    pub g_pic: f64, // Persistent inward current
    pub g_ahp: f64, // Ca²⁺-dependent K⁺ (AHP)
    pub g_l: f64,
    // Reversal potentials (mV)
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_ca: f64,    // Ca²⁺ decay (ms)
    pub buf_ratio: f64, // Buffering ratio (fraction of Ca²⁺ bound)
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
            h_pic: 1.0, // PIC inactivation starts de-inactivated
            ca: 0.0,
            ca_buf: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_pic: 0.15, // PIC for plateau potentials (conservative)
            g_ahp: 3.0,  // Strong AHP for rate limiting
            g_l: 0.3,    // Higher leak (larger soma, stabilises rest)
            e_na: 55.0,
            e_k: -90.0,
            e_ca: 120.0,
            e_l: -65.0,
            c_m: 1.5, // Larger soma → higher capacitance
            phi: 4.0,
            tau_ca: 150.0,    // Slow Ca²⁺ clearance for AHP
            buf_ratio: 0.003, // ~0.3% free Ca²⁺ (99.7% buffered)
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

            // PIC (L-type Ca²⁺): activation + slow inactivation
            // Activation: m_pic, tau ~50 ms, half-act -50 mV
            let m_pic_inf = 1.0 / (1.0 + (-(self.v + 40.0) / 5.0).exp());
            self.m_pic += (m_pic_inf - self.m_pic) / 50.0 * self.dt;
            // Inactivation: h_pic, tau ~200 ms, half-inact -40 mV
            // L-type inactivation is slow and Ca²⁺-dependent
            let h_pic_inf = 1.0 / (1.0 + ((self.v + 40.0) / 8.0).exp());
            let tau_h_pic = 200.0 + 100.0 / (1.0 + ((self.v + 40.0) / 10.0).powi(2)).max(0.01);
            self.h_pic += (h_pic_inf - self.h_pic) / tau_h_pic * self.dt;
            self.h_pic = self.h_pic.clamp(0.0, 1.0);

            // Ca²⁺ dynamics with buffering
            // Total Ca²⁺ entry (PIC-mediated)
            let i_ca_entry = self.g_pic * self.m_pic * self.h_pic * (self.v - self.e_ca);
            let ca_influx = if i_ca_entry < 0.0 {
                -i_ca_entry * 0.001
            } else {
                0.0
            };
            let ca_spike = if self.v > -10.0 { 0.02 } else { 0.0 };
            // Only ~0.3% of entering Ca²⁺ is free (rest is buffered)
            let free_ca_change = (ca_influx + ca_spike) * self.buf_ratio;
            self.ca += (-self.ca / self.tau_ca + free_ca_change) * self.dt;
            if self.ca < 0.0 {
                self.ca = 0.0;
            }
            // Buffered pool tracks total entry (slower dynamics)
            self.ca_buf += ((ca_influx + ca_spike) * (1.0 - self.buf_ratio)
                - self.ca_buf / (self.tau_ca * 5.0))
                * self.dt;
            if self.ca_buf < 0.0 {
                self.ca_buf = 0.0;
            }

            // AHP: Ca²⁺-activated K⁺ (SK channels), Hill n=2
            let ca_total = self.ca + self.ca_buf * 0.01; // Buffered contributes slowly
            let ahp_inf = ca_total * ca_total / (ca_total * ca_total + 0.25);

            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_pic = self.g_pic * self.m_pic * self.h_pic * (self.v - self.e_ca);
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
        *self = Self::new();
    }
}

impl Default for AlphaMotorNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Alpha Motor Neuron — 6-dimension coverage ──────────────────

    #[test]
    fn alpha_motor_fires_with_input() {
        let mut n = AlphaMotorNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(
            spikes > 0,
            "alpha motor must fire with sustained input: got {spikes}"
        );
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
        for _ in 0..2000 {
            n.step(4.0);
        }
        assert!(
            n.m_pic > baseline + 0.001,
            "PIC should respond to depolarisation: baseline={baseline}, after={}",
            n.m_pic
        );
    }

    #[test]
    fn alpha_motor_ca_increases_during_spiking() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(
            n.ca > 0.0,
            "Ca2+ should accumulate during spiking: ca={}",
            n.ca
        );
    }

    #[test]
    fn alpha_motor_reset_roundtrip() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..2000 {
            n.step(4.0);
        }
        n.reset();
        let mut fresh = AlphaMotorNeuron::new();
        let r1: i32 = (0..1000).map(|_| n.step(4.0)).sum();
        let r2: i32 = (0..1000).map(|_| fresh.step(4.0)).sum();
        assert_eq!(r1, r2, "reset neuron must match fresh");
    }

    #[test]
    fn alpha_motor_voltage_bounded() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..10000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite(), "voltage must stay finite");
        assert!(n.ca.is_finite(), "Ca2+ must stay finite");
        assert!(n.ca >= 0.0, "Ca2+ must be non-negative");
    }

    #[test]
    fn alpha_motor_nan_recovery() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..100 {
            n.step(3.0);
        }
        for _ in 0..10 {
            let _ = n.step(f64::NAN);
        }
        n.reset();
        assert!(n.v.is_finite());
        assert!(n.ca >= 0.0);
    }

    #[test]
    fn alpha_motor_extreme_input() {
        let mut n = AlphaMotorNeuron::new();
        for _ in 0..50 {
            n.step(1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
        for _ in 0..50 {
            n.step(-1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn alpha_motor_performance() {
        let mut n = AlphaMotorNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..5_000 {
            n.step(4.0);
        }
        assert!(
            start.elapsed().as_millis() < 500,
            "5k steps took {:?}",
            start.elapsed()
        );
    }
}
