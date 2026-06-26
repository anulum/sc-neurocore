// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hardware neuromorphic chip emulators

//! Hardware neuromorphic chip emulators.

/// Loihi CUBA LIF — Intel Loihi 1 fixed-point neuron. Davies et al. 2018.
#[derive(Clone, Debug)]
pub struct LoihiCUBANeuron {
    pub v: i32,
    pub u: i32,
    pub tau_v: i32,
    pub tau_u: i32,
    pub v_threshold: i32,
    pub v_reset: i32,
}

impl LoihiCUBANeuron {
    pub fn new() -> Self {
        Self {
            v: 0,
            u: 0,
            tau_v: 10,
            tau_u: 5,
            v_threshold: 1000,
            v_reset: 0,
        }
    }
    pub fn step(&mut self, weighted_input: i32) -> i32 {
        self.u = self.u - self.u / self.tau_u + weighted_input;
        self.v = self.v - self.v / self.tau_v + self.u;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0;
        self.u = 0;
    }
}
impl Default for LoihiCUBANeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Loihi 2 — Intel Loihi 2 three-state integer neuron.
#[derive(Clone, Debug)]
pub struct Loihi2Neuron {
    pub s1: i32,
    pub s2: i32,
    pub s3: i32,
    pub tau1: i32,
    pub tau2: i32,
    pub tau3: i32,
    pub w12: i32,
    pub w13: i32,
    pub w23: i32,
    pub s1_threshold: i32,
    pub s1_reset: i32,
    pub s3_incr: i32,
}

impl Loihi2Neuron {
    pub fn new() -> Self {
        Self {
            s1: 0,
            s2: 0,
            s3: 0,
            tau1: 10,
            tau2: 5,
            tau3: 50,
            w12: 1,
            w13: 0,
            w23: 0,
            s1_threshold: 1000,
            s1_reset: 0,
            s3_incr: 10,
        }
    }
    pub fn step(&mut self, weighted_input: i32) -> i32 {
        self.s3 -= self.s3 / self.tau3;
        self.s2 = self.s2 - self.s2 / self.tau2 + weighted_input + self.w23 * self.s3;
        self.s1 = self.s1 - self.s1 / self.tau1 + self.w12 * self.s2 + self.w13 * self.s3;
        if self.s1 >= self.s1_threshold {
            self.s1 = self.s1_reset;
            self.s3 += self.s3_incr;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.s1 = 0;
        self.s2 = 0;
        self.s3 = 0;
    }
}
impl Default for Loihi2Neuron {
    fn default() -> Self {
        Self::new()
    }
}

/// TrueNorth — IBM TrueNorth digital crossbar neuron. Merolla et al. 2014.
#[derive(Clone, Debug)]
pub struct TrueNorthNeuron {
    pub v: i32,
    pub leak: i32,
    pub threshold: i32,
    pub v_reset: i32,
}

impl TrueNorthNeuron {
    pub fn new(threshold: i32) -> Self {
        Self {
            v: 0,
            leak: 0,
            threshold,
            v_reset: 0,
        }
    }
    pub fn step(&mut self, weighted_input: i32) -> i32 {
        self.v += weighted_input - self.leak;
        if self.v >= self.threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0;
    }
}
impl Default for TrueNorthNeuron {
    fn default() -> Self {
        Self::new(100)
    }
}

/// BrainScaleS AdEx — Heidelberg analog wafer-scale. Schemmel et al. 2010.
#[derive(Clone, Debug)]
pub struct BrainScaleSAdExNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub delta_t: f64,
    pub v_rh: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub hw_speedup: f64,
    pub dt: f64,
}

impl BrainScaleSAdExNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -50.0,
            delta_t: 2.0,
            v_rh: -55.0,
            tau: 20.0,
            tau_w: 100.0,
            a: 0.5,
            b: 7.0,
            hw_speedup: 1000.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let exp_arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = (-(self.v - self.v_rest) + exp_term - self.w + current) / self.tau * self.dt;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        self.v += dv;
        self.w += dw;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.w += self.b;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}
impl Default for BrainScaleSAdExNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// SpiNNaker LIF — ARM Cortex-M4 digital LIF with refractory. Furber et al. 2014.
#[derive(Clone, Debug)]
pub struct SpiNNakerLIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub i_offset: f64,
    pub tau_refrac: f64,
    pub refrac_count: f64,
    pub dt: f64,
}

impl SpiNNakerLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            i_offset: 0.0,
            tau_refrac: 2.0,
            refrac_count: 0.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        if self.refrac_count > 0.0 {
            self.refrac_count -= self.dt;
            return 0;
        }
        self.v += (-(self.v - self.v_rest) + current + self.i_offset) / self.tau_m * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.tau_refrac;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refrac_count = 0.0;
    }
}
impl Default for SpiNNakerLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// SpiNNaker2 — TU Dresden ARM Cortex-M4F fixed-point LIF.
#[derive(Clone, Debug)]
pub struct SpiNNaker2Neuron {
    pub v: i32,
    pub v_rest: i32,
    pub v_reset: i32,
    pub v_threshold: i32,
    pub decay_mult: i32,
    pub decay_shift: i32,
    pub refrac_steps: i32,
    pub refrac_count: i32,
}

impl SpiNNaker2Neuron {
    pub fn new() -> Self {
        Self {
            v: 0,
            v_rest: 0,
            v_reset: 0,
            v_threshold: 1024,
            decay_mult: 243,
            decay_shift: 8,
            refrac_steps: 2,
            refrac_count: 0,
        }
    }
    pub fn step(&mut self, current: i32) -> i32 {
        if self.refrac_count > 0 {
            self.refrac_count -= 1;
            return 0;
        }
        self.v = ((self.v - self.v_rest).wrapping_mul(self.decay_mult) >> self.decay_shift)
            + self.v_rest
            + current;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.refrac_steps;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refrac_count = 0;
    }
}
impl Default for SpiNNaker2Neuron {
    fn default() -> Self {
        Self::new()
    }
}

/// DPI — Differential Pair Integrator (log-domain analog). Bartolozzi & Indiveri 2007.
#[derive(Clone, Debug)]
pub struct DPINeuron {
    pub i_mem: f64,
    pub i_threshold: f64,
    pub i_reset: f64,
    pub i_leak: f64,
    pub tau: f64,
    pub gain: f64,
    pub dt: f64,
}

impl DPINeuron {
    pub fn new() -> Self {
        Self {
            i_mem: 0.0,
            i_threshold: 1.0,
            i_reset: 0.0,
            i_leak: 0.01,
            tau: 20.0,
            gain: 1.0,
            dt: 1.0,
        }
    }
    pub fn step(&mut self, i_syn: f64) -> i32 {
        self.i_mem += (-self.i_mem + self.gain * i_syn + self.i_leak) / self.tau * self.dt;
        if self.i_mem >= self.i_threshold {
            self.i_mem = self.i_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.i_mem = 0.0;
    }
}
impl Default for DPINeuron {
    fn default() -> Self {
        Self::new()
    }
}

/// Akida — BrainChip event-domain rank-order neuron.
#[derive(Clone, Debug)]
pub struct AkidaNeuron {
    pub v: i32,
    pub threshold: i32,
    pub modulation: f64,
    pub rank: i32,
    pub spiked: bool,
}

impl AkidaNeuron {
    pub fn new(threshold: i32) -> Self {
        Self {
            v: 0,
            threshold,
            modulation: 0.75,
            rank: 0,
            spiked: false,
        }
    }
    pub fn step(&mut self, weight: f64) -> i32 {
        if self.spiked {
            return 0;
        }
        self.v += (weight * self.modulation.powi(self.rank)) as i32;
        self.rank += 1;
        if self.v >= self.threshold {
            self.spiked = true;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0;
        self.rank = 0;
        self.spiked = false;
    }
}
impl Default for AkidaNeuron {
    fn default() -> Self {
        Self::new(100)
    }
}

/// NeuroGrid — Boahen 2014 subthreshold analog 2-compartment.
#[derive(Clone, Debug)]
pub struct NeuroGridNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub tau_s: f64,
    pub tau_d: f64,
    pub g_c: f64,
    pub delta_t: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub v_peak: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl NeuroGridNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -65.0,
            v_d: -65.0,
            tau_s: 20.0,
            tau_d: 50.0,
            g_c: 0.5,
            delta_t: 2.0,
            v_rest: -65.0,
            v_threshold: -50.0,
            v_peak: 20.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    fn valid(&self) -> bool {
        self.v_s.is_finite()
            && self.v_d.is_finite()
            && self.tau_s.is_finite()
            && self.tau_s > 0.0
            && self.tau_d.is_finite()
            && self.tau_d > 0.0
            && self.g_c.is_finite()
            && self.g_c >= 0.0
            && self.delta_t.is_finite()
            && self.delta_t > 0.0
            && self.v_rest.is_finite()
            && self.v_threshold.is_finite()
            && self.v_peak.is_finite()
            && self.v_reset.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
    }

    fn derivatives(&self, v_s: f64, v_d: f64, current: f64) -> (f64, f64) {
        let v_s_eff = v_s.min(self.v_peak);
        let dv_d = (-(v_d - self.v_rest) + current - self.g_c * (v_d - v_s_eff)) / self.tau_d;
        let exp_arg = ((v_s_eff - self.v_threshold) / self.delta_t).min(20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv_s = (-(v_s_eff - self.v_rest) + exp_term + self.g_c * (v_d - v_s_eff)) / self.tau_s;
        (dv_s, dv_d)
    }

    fn rk4_substep(&self, v_s: f64, v_d: f64, current: f64) -> (f64, f64) {
        let dt = self.dt;
        let (k1s, k1d) = self.derivatives(v_s, v_d, current);
        let (k2s, k2d) = self.derivatives(v_s + 0.5 * dt * k1s, v_d + 0.5 * dt * k1d, current);
        let (k3s, k3d) = self.derivatives(v_s + 0.5 * dt * k2s, v_d + 0.5 * dt * k2d, current);
        let (k4s, k4d) = self.derivatives(v_s + dt * k3s, v_d + dt * k3d, current);
        (
            v_s + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0,
            v_d + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0,
        )
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid() {
            return 0;
        }
        let (next_v_s, next_v_d) = self.rk4_substep(self.v_s, self.v_d, current);
        if !next_v_s.is_finite() || !next_v_d.is_finite() {
            return 0;
        }
        self.v_d = next_v_d;
        if next_v_s >= self.v_peak {
            self.v_s = self.v_reset;
            1
        } else {
            self.v_s = next_v_s;
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -65.0;
        self.v_d = -65.0;
    }
}
impl Default for NeuroGridNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loihi_cuba_fires() {
        let mut n = LoihiCUBANeuron::new();
        let t: i32 = (0..200).map(|_| n.step(100)).sum();
        assert!(t > 0);
    }
    #[test]
    fn loihi2_fires() {
        let mut n = Loihi2Neuron {
            tau3: 8,
            ..Loihi2Neuron::new()
        };
        let t: i32 = (0..500).map(|_| n.step(200)).sum();
        assert!(t > 0);
    }
    #[test]
    fn truenorth_fires() {
        let mut n = TrueNorthNeuron::default();
        let t: i32 = (0..10).map(|_| n.step(50)).sum();
        assert!(t > 0);
    }
    #[test]
    fn brainscales_fires() {
        let mut n = BrainScaleSAdExNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(500.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn spinnaker_fires() {
        let mut n = SpiNNakerLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(30.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn spinnaker2_fires() {
        let mut n = SpiNNaker2Neuron::new();
        let t: i32 = (0..200).map(|_| n.step(100)).sum();
        assert!(t > 0);
    }
    #[test]
    fn dpi_fires() {
        let mut n = DPINeuron::new();
        let t: i32 = (0..100).map(|_| n.step(1.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn akida_fires() {
        let mut n = AkidaNeuron::default();
        let t: i32 = (0..10).map(|_| n.step(50.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn neurogrid_fires() {
        let mut n = NeuroGridNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(500.0)).sum();
        assert!(t > 0);
    }

    // ── Multi-angle tests for hardware models ──

    // -- LoihiCUBA --
    #[test]
    fn loihi_cuba_silent() {
        let mut n = LoihiCUBANeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn loihi_cuba_reset() {
        let mut n = LoihiCUBANeuron::new();
        for _ in 0..50 {
            n.step(100);
        }
        n.reset();
        assert_eq!(n.v, 0);
        assert_eq!(n.u, 0);
    }
    #[test]
    fn loihi_cuba_bounded() {
        let mut n = LoihiCUBANeuron::new();
        for _ in 0..1000 {
            n.step(10000);
        }
    }

    // -- Loihi2 --
    #[test]
    fn loihi2_silent() {
        let mut n = Loihi2Neuron::new();
        let t: i32 = (0..200).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn loihi2_reset() {
        let mut n = Loihi2Neuron::new();
        for _ in 0..50 {
            n.step(200);
        }
        n.reset();
        assert_eq!(n.s1, 0);
    }
    #[test]
    fn loihi2_bounded() {
        let mut n = Loihi2Neuron {
            tau3: 8,
            ..Loihi2Neuron::new()
        };
        for _ in 0..1000 {
            n.step(10000);
        }
    }

    // -- TrueNorth --
    #[test]
    fn truenorth_silent() {
        let mut n = TrueNorthNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn truenorth_reset() {
        let mut n = TrueNorthNeuron::default();
        for _ in 0..10 {
            n.step(50);
        }
        n.reset();
        assert_eq!(n.v, 0);
    }

    // -- BrainScaleSAdEx --
    #[test]
    fn brainscales_silent() {
        let mut n = BrainScaleSAdExNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn brainscales_reset() {
        let mut n = BrainScaleSAdExNeuron::new();
        for _ in 0..100 {
            n.step(500.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn brainscales_bounded() {
        let mut n = BrainScaleSAdExNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn brainscales_nan_no_panic() {
        BrainScaleSAdExNeuron::new().step(f64::NAN);
    }

    // -- SpiNNakerLIF --
    #[test]
    fn spinnaker_silent() {
        let mut n = SpiNNakerLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn spinnaker_reset() {
        let mut n = SpiNNakerLIFNeuron::new();
        for _ in 0..50 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn spinnaker_bounded() {
        let mut n = SpiNNakerLIFNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn spinnaker_nan_no_panic() {
        SpiNNakerLIFNeuron::new().step(f64::NAN);
    }

    // -- SpiNNaker2 --
    #[test]
    fn spinnaker2_silent() {
        let mut n = SpiNNaker2Neuron::new();
        let t: i32 = (0..200).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn spinnaker2_reset() {
        let mut n = SpiNNaker2Neuron::new();
        for _ in 0..50 {
            n.step(100);
        }
        n.reset();
    }
    #[test]
    fn spinnaker2_bounded() {
        let mut n = SpiNNaker2Neuron::new();
        for _ in 0..1000 {
            n.step(10000);
        }
    }

    // -- DPI --
    #[test]
    fn dpi_silent() {
        let mut n = DPINeuron::new();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn dpi_reset() {
        let mut n = DPINeuron::new();
        for _ in 0..50 {
            n.step(1.0);
        }
        n.reset();
    }
    #[test]
    fn dpi_nan_no_panic() {
        DPINeuron::new().step(f64::NAN);
    }

    // -- Akida --
    #[test]
    fn akida_silent() {
        let mut n = AkidaNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn akida_reset() {
        let mut n = AkidaNeuron::default();
        for _ in 0..10 {
            n.step(50.0);
        }
        n.reset();
    }
    #[test]
    fn akida_nan_no_panic() {
        AkidaNeuron::default().step(f64::NAN);
    }

    // -- NeuroGrid --
    #[test]
    fn neurogrid_silent() {
        let mut n = NeuroGridNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn neurogrid_reset() {
        let mut n = NeuroGridNeuron::new();
        for _ in 0..100 {
            n.step(500.0);
        }
        n.reset();
        assert!((n.v_s - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn neurogrid_bounded() {
        let mut n = NeuroGridNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v_s.is_finite());
    }
    #[test]
    fn neurogrid_nan_no_panic() {
        NeuroGridNeuron::new().step(f64::NAN);
    }
    #[test]
    fn neurogrid_rk4_anchor() {
        let mut n = NeuroGridNeuron::new();
        let spikes: i32 = (0..20_000).map(|_| n.step(100.0)).sum();
        assert_eq!(spikes, 94);
        assert!(n.v_s.is_finite());
        assert!(n.v_d.is_finite());
    }
    #[test]
    fn neurogrid_invalid_input_preserves_state() {
        let mut n = NeuroGridNeuron::new();
        for _ in 0..10 {
            n.step(100.0);
        }
        let old = (n.v_s, n.v_d);
        assert_eq!(n.step(f64::INFINITY), 0);
        assert_eq!((n.v_s, n.v_d), old);
    }
}
