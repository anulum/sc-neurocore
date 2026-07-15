// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Chay-Keizer Neuron Model

//! Chay-Keizer pancreatic beta-cell dynamics.

/// Chay-Keizer — modified beta cell with inactivating Ca current.
#[derive(Clone, Debug)]
pub struct ChayKeizerNeuron {
    pub v: f64,
    pub n: f64,
    pub ca: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub k_d: f64,
    pub f_ca: f64,
    pub k_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ChayKeizerNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.01,
            ca: 0.1,
            g_ca: 20.0,
            g_k: 25.0,
            g_kca: 12.0,
            g_l: 0.1,
            e_ca: 100.0,
            e_k: -75.0,
            e_l: -40.0,
            k_d: 1.0,
            f_ca: 0.004,
            k_ca: 0.03,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 25.0) / 8.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 18.0) / 14.0).exp());
        let tau_n = (20.0 / (1.0 + ((self.v + 18.0) / 14.0).exp())).max(0.1);
        let q_kca = self.ca / (self.ca + self.k_d);
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.n * (self.v - self.e_k);
        let i_kca = self.g_kca * q_kca * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_ca - i_k - i_kca - i_l + current) * self.dt;
        self.v = self.v.clamp(-200.0, 200.0);
        self.n += (n_inf - self.n) / tau_n * self.dt;
        self.ca = (self.ca + (-self.f_ca * i_ca - self.k_ca * self.ca) * self.dt).max(0.0);
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.01;
        self.ca = 0.1;
    }
}
impl Default for ChayKeizerNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = ChayKeizerNeuron::default();
        let constructed = ChayKeizerNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn chay_keizer_fires() {
        let mut n = ChayKeizerNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn chay_keizer_reset_clears_state() {
        let mut n = ChayKeizerNeuron::new();
        for _ in 0..1000 {
            n.step(10.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn chay_keizer_bounded() {
        let mut n = ChayKeizerNeuron::new();
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn chay_keizer_nan_no_panic() {
        ChayKeizerNeuron::new().step(f64::NAN);
    }

    #[test]
    fn chay_keizer_negative_no_crash() {
        let mut n = ChayKeizerNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
}
