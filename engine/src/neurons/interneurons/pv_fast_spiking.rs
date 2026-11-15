// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PV fast-spiking interneuron model

#![deny(missing_docs)]

use super::super::biophysical::safe_rate;

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
    /// Membrane potential (mV).
    pub v: f64,
    /// Sodium inactivation gate (dimensionless); Wang-Buzsáki kinetics scaled by `phi`.
    pub h: f64,
    /// Delayed-rectifier potassium activation gate (dimensionless); kinetics scaled by
    /// `phi`.
    pub n: f64,
    /// Kv3.1 activation gate (dimensionless).
    pub p: f64,
    // Conductances (mS/cm²)
    /// Maximal sodium conductance (mS/cm^2).
    pub g_na: f64,
    /// Maximal delayed-rectifier potassium conductance (mS/cm^2).
    pub g_k: f64,
    /// Maximal Kv3-type potassium conductance (mS/cm^2).
    pub g_kv3: f64,
    /// Maximal leak conductance (mS/cm^2).
    pub g_l: f64,
    // Reversal potentials (mV)
    /// Sodium reversal potential (mV).
    pub e_na: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// Membrane capacitance (uF/cm^2).
    pub c_m: f64,
    /// Temperature scaling factor of the gating kinetics (dimensionless).
    pub phi: f64,
    /// RK4 sub-step (ms). Each call runs `floor(0.5 / max(dt, 0.001))` sub-steps, about
    /// 0.5 ms; a `dt` above 0.5 ms advances nothing.
    pub dt: f64,
    /// Voltage at which a spike is registered (mV).
    pub v_threshold: f64,
}

impl PVFastSpikingNeuron {
    /// Create a cell at -65 mV with the default gates and parameters and `dt` = 0.01 ms.
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

    /// Advance about 0.5 ms of classical RK4 sub-steps (see `dt`) with `current` (uA/cm^2)
    /// held constant; return 1 when `v` crosses `v_threshold` upwards between the start and
    /// the end of the call, else 0.
    ///
    /// The four states are integrated together. The Kv3.1 gate relaxes towards
    /// `1 / (1 + exp(-(v + 10) / 10))` at rate `phi` per ms. `v` is never reset. The input
    /// is not validated.
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

    /// Restore the four states to their initial values; the parameters are kept.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
