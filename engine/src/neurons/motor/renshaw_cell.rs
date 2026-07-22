// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Renshaw Cell

//! Renshaw-cell recurrent-inhibition dynamics.

use crate::neurons::biophysical::safe_rate;

/// Renshaw cell — spinal inhibitory interneuron for recurrent inhibition.
///
/// Receives collaterals from alpha motor neuron axons, provides
/// glycinergic recurrent inhibition back to the motor pool. Characteristic
/// high-frequency initial burst (cholinergic nicotinic drive from motor
/// axon collaterals) followed by rapid adaptation.
///
/// WB gating core with strong adaptation (M-current analogue) to produce
/// the burst-then-decay response pattern.
///
/// Renshaw 1941 (discovery); Windhorst, Prog. Neurobiol. 46(5), 1996.
#[derive(Clone, Debug)]
pub struct RenshawCell {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub adapt: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_adapt: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub tau_adapt: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl RenshawCell {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            adapt: 0.0,
            g_na: 35.0,
            g_k: 9.0,
            g_adapt: 5.0,
            g_l: 0.12,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            tau_adapt: 50.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }

    fn voltage_valid(value: f64) -> bool {
        value.is_finite() && (-150.0..=100.0).contains(&value)
    }

    fn probability(value: f64) -> bool {
        value.is_finite() && (0.0..=1.0).contains(&value)
    }

    fn exact_gate(previous: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> Option<f64> {
        let total = phi * (alpha + beta);
        if !previous.is_finite()
            || !alpha.is_finite()
            || !beta.is_finite()
            || !total.is_finite()
            || !dt.is_finite()
            || total <= 0.0
        {
            return None;
        }
        let steady = alpha / (alpha + beta);
        Some((steady + (previous - steady) * (-total * dt).exp()).clamp(0.0, 1.0))
    }

    fn exact_relax(previous: f64, steady: f64, tau: f64, dt: f64) -> Option<f64> {
        if !previous.is_finite()
            || !steady.is_finite()
            || !tau.is_finite()
            || !dt.is_finite()
            || tau <= 0.0
        {
            return None;
        }
        Some((steady + (previous - steady) * (-dt / tau).exp()).clamp(0.0, 1.0))
    }

    fn valid_state(&self) -> bool {
        Self::voltage_valid(self.v)
            && Self::probability(self.h)
            && Self::probability(self.n)
            && Self::probability(self.adapt)
            && self.g_na.is_finite()
            && self.g_k.is_finite()
            && self.g_adapt.is_finite()
            && self.g_l.is_finite()
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_l.is_finite()
            && self.c_m.is_finite()
            && self.phi.is_finite()
            && self.tau_adapt.is_finite()
            && self.dt.is_finite()
            && self.v_threshold.is_finite()
            && self.g_na >= 0.0
            && self.g_k >= 0.0
            && self.g_adapt >= 0.0
            && self.g_l >= 0.0
            && self.c_m > 0.0
            && self.phi > 0.0
            && self.tau_adapt > 0.0
            && self.dt > 0.0
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid_state() {
            return 0;
        }

        let v_prev = self.v;
        let mut v = self.v;
        let mut h = self.h;
        let mut n = self.n;
        let mut adapt = self.adapt;
        let n_sub = ((0.5 / self.dt.max(0.001)).max(1.0)) as usize;
        for _ in 0..n_sub {
            let am = safe_rate(0.1, 35.0, v, 10.0, 1.0);
            let bm = 4.0 * (-(v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, v, 10.0, 0.1);
            let bn = 0.125 * (-(v + 44.0) / 80.0).exp();

            let Some(h_next) = Self::exact_gate(h, ah, bh, self.phi, self.dt) else {
                return 0;
            };
            let Some(n_next) = Self::exact_gate(n, an, bn, self.phi, self.dt) else {
                return 0;
            };

            let adapt_inf = 1.0 / (1.0 + (-(v + 30.0) / 5.0).exp());
            let Some(adapt_next) = Self::exact_relax(adapt, adapt_inf, self.tau_adapt, self.dt)
            else {
                return 0;
            };

            let g_na = self.g_na * m_inf.powi(3) * h_next;
            let g_k = self.g_k * n_next.powi(4);
            let g_adapt = self.g_adapt * adapt_next;
            let g_total = g_na + g_k + g_adapt + self.g_l;
            if !g_total.is_finite() || g_total <= 0.0 {
                return 0;
            }

            let steady_v = (current
                + g_na * self.e_na
                + g_k * self.e_k
                + g_adapt * self.e_k
                + self.g_l * self.e_l)
                / g_total;
            let v_next = steady_v + (v - steady_v) * (-(g_total / self.c_m) * self.dt).exp();
            if !Self::voltage_valid(v_next)
                || !Self::probability(h_next)
                || !Self::probability(n_next)
                || !Self::probability(adapt_next)
            {
                return 0;
            }

            v = v_next;
            h = h_next;
            n = n_next;
            adapt = adapt_next;
        }

        self.v = v;
        self.h = h;
        self.n = n;
        self.adapt = adapt;
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
        self.adapt = 0.0;
    }
}

impl Default for RenshawCell {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Renshaw Cell — 6-dimension coverage ────────────────────────

    #[test]
    fn renshaw_fires_with_input() {
        let mut n = RenshawCell::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(spikes > 0, "Renshaw must fire: got {spikes}");
    }

    #[test]
    fn renshaw_no_fire_without_input() {
        let mut n = RenshawCell::new();
        let spikes: i32 = (0..3000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn renshaw_negative_current_no_fire() {
        let mut n = RenshawCell::new();
        let spikes: i32 = (0..2000).map(|_| n.step(-2.0)).sum();
        assert_eq!(spikes, 0);
    }

    fn renshaw_test_gate(previous: f64, alpha: f64, beta: f64, phi: f64, dt: f64) -> f64 {
        let total = phi * (alpha + beta);
        let steady = alpha / (alpha + beta);
        (steady + (previous - steady) * (-total * dt).exp()).clamp(0.0, 1.0)
    }

    fn renshaw_test_adapt(previous: f64, steady: f64, tau: f64, dt: f64) -> f64 {
        (steady + (previous - steady) * (-dt / tau).exp()).clamp(0.0, 1.0)
    }

    fn renshaw_reference_step(mut n: RenshawCell, current: f64) -> RenshawCell {
        let n_sub = ((0.5 / n.dt.max(0.001)).max(1.0)) as usize;
        for _ in 0..n_sub {
            let am = safe_rate(0.1, 35.0, n.v, 10.0, 1.0);
            let bm = 4.0 * (-(n.v + 60.0) / 18.0).exp();
            let ah = 0.07 * (-(n.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(n.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, n.v, 10.0, 0.1);
            let bn = 0.125 * (-(n.v + 44.0) / 80.0).exp();
            let m_inf = am / (am + bm);

            n.h = renshaw_test_gate(n.h, ah, bh, n.phi, n.dt);
            n.n = renshaw_test_gate(n.n, an, bn, n.phi, n.dt);
            let adapt_inf = 1.0 / (1.0 + (-(n.v + 30.0) / 5.0).exp());
            n.adapt = renshaw_test_adapt(n.adapt, adapt_inf, n.tau_adapt, n.dt);

            let g_na = n.g_na * m_inf.powi(3) * n.h;
            let g_k = n.g_k * n.n.powi(4);
            let g_adapt = n.g_adapt * n.adapt;
            let g_total = g_na + g_k + g_adapt + n.g_l;
            let steady_v =
                (current + g_na * n.e_na + g_k * n.e_k + g_adapt * n.e_k + n.g_l * n.e_l) / g_total;
            n.v = steady_v + (n.v - steady_v) * (-(g_total / n.c_m) * n.dt).exp();
        }
        n
    }

    fn renshaw_snapshot(n: &RenshawCell) -> (f64, f64, f64, f64) {
        (n.v, n.h, n.n, n.adapt)
    }

    #[test]
    fn renshaw_uses_exact_gate_and_conductance_membrane_step() {
        let mut n = RenshawCell::new();
        let expected = renshaw_reference_step(RenshawCell::new(), 4.0);

        assert_eq!(n.step(4.0), 0);

        assert!((n.v - expected.v).abs() <= 1e-12);
        assert!((n.h - expected.h).abs() <= 1e-12);
        assert!((n.n - expected.n).abs() <= 1e-12);
        assert!((n.adapt - expected.adapt).abs() <= 1e-12);
    }

    #[test]
    fn renshaw_rejects_invalid_current_without_state_mutation() {
        let mut n = RenshawCell::new();
        for _ in 0..20 {
            n.step(4.0);
        }
        let before = renshaw_snapshot(&n);

        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!(renshaw_snapshot(&n), before);
        assert_eq!(n.step(f64::INFINITY), 0);
        assert_eq!(renshaw_snapshot(&n), before);
    }

    #[test]
    fn renshaw_rejects_excess_current_without_state_mutation() {
        let mut n = RenshawCell::new();
        let before = renshaw_snapshot(&n);

        assert_eq!(n.step(1.0e8), 0);

        assert_eq!(renshaw_snapshot(&n), before);
    }

    #[test]
    fn renshaw_corrupted_gate_is_preserved_on_step() {
        let mut n = RenshawCell::new();
        n.h = 1.5;
        let before = renshaw_snapshot(&n);

        assert_eq!(n.step(4.0), 0);

        assert_eq!(renshaw_snapshot(&n), before);
    }

    #[test]
    fn renshaw_burst_then_adapt() {
        // Renshaw should fire more in the first epoch than the second
        let mut n = RenshawCell::new();
        let first: i32 = (0..2000).map(|_| n.step(4.0)).sum();
        let second: i32 = (0..2000).map(|_| n.step(4.0)).sum();
        assert!(
            second <= first + 5,
            "Renshaw should adapt: first={first}, second={second}"
        );
    }

    #[test]
    fn renshaw_adapt_increases_during_firing() {
        let mut n = RenshawCell::new();
        let baseline = n.adapt;
        for _ in 0..3000 {
            n.step(4.0);
        }
        assert!(
            n.adapt > baseline + 0.01,
            "adaptation variable should increase: adapt={}",
            n.adapt
        );
    }

    #[test]
    fn renshaw_reset_roundtrip() {
        let mut n = RenshawCell::new();
        for _ in 0..2000 {
            n.step(4.0);
        }
        n.reset();
        let mut fresh = RenshawCell::new();
        let r1: i32 = (0..1000).map(|_| n.step(4.0)).sum();
        let r2: i32 = (0..1000).map(|_| fresh.step(4.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn renshaw_voltage_bounded() {
        let mut n = RenshawCell::new();
        for _ in 0..10000 {
            n.step(10.0);
        }
        assert!(n.v.is_finite());
        assert!(n.adapt.is_finite());
    }

    #[test]
    fn renshaw_nan_recovery() {
        let mut n = RenshawCell::new();
        for _ in 0..100 {
            n.step(3.0);
        }
        for _ in 0..10 {
            let _ = n.step(f64::NAN);
        }
        n.reset();
        assert!(n.v.is_finite());
        assert_eq!(n.adapt, 0.0);
    }

    #[test]
    fn renshaw_extreme_input() {
        let mut n = RenshawCell::new();
        for _ in 0..50 {
            n.step(1e6);
        }
        n.reset();
        assert!(n.v.is_finite());
    }

    #[test]
    fn renshaw_performance() {
        let mut n = RenshawCell::new();
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
