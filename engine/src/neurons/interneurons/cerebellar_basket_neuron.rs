// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Cerebellar basket-cell model

use super::super::biophysical::safe_rate;

/// Cerebellar basket cell — perisomatic-targeting interneuron.
///
/// Biophysics: Wang-Buzsáki core + A-type K+ (transient outward) +
/// Ca2+-dependent K+ (afterhyperpolarisation). Distinct from cortical
/// PV+ by A-current and pronounced AHP from Ca2+-activated K+.
///
/// Based on Midtgaard 1992 / Häusser & Clark 1997 / WB 1996.
#[derive(Clone, Debug)]
pub struct CerebellarBasketNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub a: f64,
    pub b: f64,
    pub ca: f64, // Intracellular [Ca2+] (µM)
    // Conductances
    pub g_na: f64,
    pub g_k: f64,
    pub g_a: f64,
    pub g_kca: f64,
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

impl CerebellarBasketNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            a: 0.0,
            b: 0.9,
            ca: 0.05,
            g_na: 35.0,
            g_k: 9.0,
            g_a: 3.0,
            g_kca: 2.0,
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
            // WB gating for Na+ and Kdr
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();

            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;

            // A-type K+ (cerebellar)
            let a_inf = 1.0 / (1.0 + (-(self.v + 45.0) / 15.0).exp());
            let b_inf = 1.0 / (1.0 + ((self.v + 75.0) / 8.0).exp());
            self.a += self.phi * (a_inf - self.a) / 5.0 * self.dt;
            self.b += (b_inf - self.b) / 50.0 * self.dt;

            // Ca2+-activated K+ (AHP)
            let q_inf = self.ca / (self.ca + 0.2);

            // Ca2+ dynamics: entry during depolarisation
            let i_ca_entry = if self.v > -20.0 {
                0.01 * (self.v + 20.0)
            } else {
                0.0
            };
            self.ca += (-self.ca / 80.0 + i_ca_entry) * self.dt;
            if self.ca < 0.0 {
                self.ca = 0.0;
            }

            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_a = self.g_a * self.a.powi(3) * self.b * (self.v - self.e_k);
            let i_kca = self.g_kca * q_inf * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);

            self.v += (-i_na - i_k - i_a - i_kca - i_l + current) / self.c_m * self.dt;
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
        self.a = 0.0;
        self.b = 0.9;
        self.ca = 0.05;
    }
}

impl Default for CerebellarBasketNeuron {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════
// Martinotti Cell
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basket_fires_with_input() {
        let mut n = CerebellarBasketNeuron::new();
        let spikes: i32 = (0..5000).map(|_| n.step(3.0)).sum();
        assert!(spikes > 0, "Basket cell must fire with sustained input");
    }

    #[test]
    fn basket_no_fire_without_input() {
        let mut n = CerebellarBasketNeuron::new();
        let spikes: i32 = (0..2000).map(|_| n.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn basket_ca_dynamics_during_spiking() {
        // Ca2+ decays between spikes but spikes cause transient increases
        let mut n = CerebellarBasketNeuron::new();
        // Run until steady-state Ca2+ with spiking
        for _ in 0..5000 {
            n.step(3.0);
        }
        let ca_spiking = n.ca;
        // Ca2+ without spiking should be lower (pure decay)
        let mut n2 = CerebellarBasketNeuron::new();
        n2.ca = ca_spiking;
        for _ in 0..5000 {
            n2.step(0.0);
        }
        assert!(
            ca_spiking > n2.ca,
            "spiking Ca ({ca_spiking:.4}) should exceed resting Ca ({:.4})",
            n2.ca
        );
    }

    #[test]
    fn basket_reset_roundtrip() {
        let mut n = CerebellarBasketNeuron::new();
        for _ in 0..2000 {
            n.step(3.0);
        }
        n.reset();
        assert_eq!(n.ca, 0.05);
        let mut fresh = CerebellarBasketNeuron::new();
        let r1: i32 = (0..1000).map(|_| n.step(3.0)).sum();
        let r2: i32 = (0..1000).map(|_| fresh.step(3.0)).sum();
        assert_eq!(r1, r2);
    }

    #[test]
    fn basket_voltage_bounded() {
        let mut n = CerebellarBasketNeuron::new();
        for _ in 0..5000 {
            n.step(5.0);
        }
        assert!(n.v.is_finite());
        assert!(n.ca.is_finite());
        assert!(n.ca >= 0.0);
    }

    #[test]
    #[ignore = "wall-clock performance smoke; use Criterion benches for timing evidence"]
    fn basket_performance_5k_steps() {
        let mut n = CerebellarBasketNeuron::new();
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
