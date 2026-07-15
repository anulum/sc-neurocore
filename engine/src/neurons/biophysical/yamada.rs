// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Yamada Neuron Model

//! Yamada subcritical Hopf-burster dynamics.

/// Yamada 1989 — subcritical Hopf burster.
#[derive(Clone, Debug)]
pub struct YamadaNeuron {
    pub v: f64,
    pub n: f64,
    pub q: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_q: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_q: f64,
    pub e_l: f64,
    pub tau_q: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl YamadaNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            n: 0.1,
            q: 0.0,
            g_na: 20.0,
            g_k: 10.0,
            g_q: 5.0,
            g_l: 0.5,
            e_na: 60.0,
            e_k: -80.0,
            e_q: -80.0,
            e_l: -60.0,
            tau_q: 300.0,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 30.0) / 9.5).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 30.0) / 10.0).exp());
        let q_inf = 1.0 / (1.0 + (-(self.v + 50.0) / 10.0).exp());
        let tau_n = 1.0 + 7.5 / (1.0 + ((self.v + 40.0) / 12.0).exp());
        let i_na = self.g_na * m_inf.powi(3) * (1.0 - self.n) * (self.v - self.e_na);
        let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
        let i_q = self.g_q * self.q * (self.v - self.e_q);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_na - i_k - i_q - i_l + current) * self.dt;
        self.n += (n_inf - self.n) / tau_n * self.dt;
        self.q += (q_inf - self.q) / self.tau_q * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -60.0;
        self.n = 0.1;
        self.q = 0.0;
    }
}
impl Default for YamadaNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = YamadaNeuron::default();
        let constructed = YamadaNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn yamada_fires() {
        let mut n = YamadaNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    // -- Yamada --
    #[test]
    fn yamada_silent_without_input() {
        let mut n = YamadaNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn yamada_reset_clears_state() {
        let mut n = YamadaNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-60.0)).abs() < 1e-10);
        assert!((n.q - 0.0).abs() < 1e-10);
    }
    #[test]
    fn yamada_extreme_bounded() {
        let mut n = YamadaNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn yamada_slow_q_adapts() {
        let mut n = YamadaNeuron::new();
        for _ in 0..2000 {
            n.step(5.0);
        }
        assert!(n.q > 0.0, "slow variable q should activate during spiking");
    }
    #[test]
    fn yamada_negative_no_crash() {
        let mut n = YamadaNeuron::new();
        for _ in 0..200 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn yamada_nan_no_panic() {
        let mut n = YamadaNeuron::new();
        n.step(f64::NAN);
    }
}
