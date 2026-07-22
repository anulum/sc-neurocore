// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SST interneuron model

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
