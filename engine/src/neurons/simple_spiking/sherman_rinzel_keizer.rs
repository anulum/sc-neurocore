// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sherman-Rinzel-Keizer Neuron Model

//! Sherman-Rinzel-Keizer pancreatic beta-cell dynamics.

/// Sherman-Rinzel-Keizer 1988 — pancreatic beta cell (reduced).
#[derive(Clone, Debug)]
pub struct ShermanRinzelKeizerNeuron {
    pub v: f64,
    pub n: f64,
    pub s: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_s: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub tau_s: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ShermanRinzelKeizerNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.1,
            s: 0.1,
            g_ca: 3.6,
            g_k: 10.0,
            g_s: 4.0,
            e_ca: 25.0,
            e_k: -75.0,
            tau_s: 5000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v + 20.0) / 12.0).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v + 16.0) / 5.0).exp());
        let s_inf = 1.0 / (1.0 + (-(self.v + 35.0) / 10.0).exp());
        let tau_n = 9.09;
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * self.n * (self.v - self.e_k);
        let i_s = self.g_s * self.s * (self.v - self.e_k);
        self.v += (-i_ca - i_k - i_s + current) * self.dt;
        self.n += (n_inf - self.n) / tau_n * self.dt;
        self.s += (s_inf - self.s) / self.tau_s * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.1;
        self.s = 0.1;
    }
}
impl Default for ShermanRinzelKeizerNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = ShermanRinzelKeizerNeuron::default();
        let constructed = ShermanRinzelKeizerNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn srk_fires() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn srk_reset_clears_state() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..1000 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn srk_bounded() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..5000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn srk_nan_no_panic() {
        ShermanRinzelKeizerNeuron::new().step(f64::NAN);
    }

    #[test]
    fn srk_negative_no_crash() {
        let mut n = ShermanRinzelKeizerNeuron::new();
        for _ in 0..500 {
            n.step(-5.0);
        }
        assert!(n.v.is_finite());
    }
}
