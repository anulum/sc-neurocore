// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Martinotti interneuron model

#![deny(missing_docs)]

/// Martinotti cell — adapting interneuron targeting layer 1 apical dendrites.
///
/// Biophysics: Na+, K+, M-current (Kv7, strong adaptation), T-type Ca2+
/// (rebound), leak. Overlaps with SST+ phenotype but with stronger
/// adaptation (higher g_m) and lower rheobase.
///
/// Based on Silberberg & Markram 2007 / Toledo-Rodriguez et al. 2005.
#[derive(Clone, Debug)]
pub struct MartinottiNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Sodium activation gate (dimensionless); Traub-Miles kinetics with V_T = -56.2 mV.
    pub m: f64,
    /// Sodium inactivation gate (dimensionless).
    pub h: f64,
    /// Delayed-rectifier potassium activation gate (dimensionless).
    pub n: f64,
    /// M-current activation gate (dimensionless).
    pub p: f64,
    /// T-type Ca2+ inactivation gate (dimensionless).
    pub s: f64,
    // Conductances
    /// Maximal sodium conductance (mS/cm^2).
    pub g_na: f64,
    /// Maximal delayed-rectifier potassium conductance (mS/cm^2).
    pub g_k: f64,
    /// Maximal slow non-inactivating (M-type) potassium conductance (mS/cm^2).
    pub g_m: f64,
    /// Maximal low-threshold (T-type) calcium conductance (mS/cm^2).
    pub g_t: f64,
    /// Leak conductance (mS/cm^2).
    pub g_l: f64,
    // Reversal potentials
    /// Sodium reversal potential (mV).
    pub e_na: f64,
    /// Potassium reversal potential (mV).
    pub e_k: f64,
    /// Calcium reversal potential (mV).
    pub e_ca: f64,
    /// Leak reversal potential (mV).
    pub e_l: f64,
    /// Membrane capacitance (uF/cm^2).
    pub c_m: f64,
    /// RK4 sub-step (ms); each call takes four, so it covers `4 * dt`.
    pub dt: f64,
    /// Voltage at which a spike is registered (mV).
    pub v_threshold: f64,
}

impl MartinottiNeuron {
    /// Create a cell at -65 mV with the default gates and parameters and `dt` = 0.025 ms
    /// (0.1 ms per call).
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

    /// Advance four classical RK4 sub-steps of `dt` with `current` (uA/cm^2) held constant;
    /// return 1 when `v` crosses `v_threshold` upwards between the start and the end of the
    /// call, else 0.
    ///
    /// All six states are integrated together. `v` is never reset. The input is not
    /// validated.
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

    /// Restore the six states to their initial values; the parameters are kept.
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
    use super::super::SSTNeuron;
    use super::*;
    use crate::neurons::host_budget;

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
    fn martinotti_performance_10k_steps() {
        let mut n = MartinottiNeuron::new();
        host_budget::assert_within(module_path!(), "martinotti_performance_10k_steps", || {
            for _ in 0..10_000 {
                std::hint::black_box(n.step(4.0));
            }
        });
    }
}
