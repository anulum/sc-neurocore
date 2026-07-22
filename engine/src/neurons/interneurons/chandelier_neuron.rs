// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Chandelier interneuron model

use super::super::biophysical::safe_rate;

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

#[cfg(test)]
mod tests {
    use super::super::PVFastSpikingNeuron;
    use super::*;

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
}
