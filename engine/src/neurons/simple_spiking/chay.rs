// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Chay Neuron Model

//! Chay pancreatic beta-cell dynamics.

/// Chay 1985 — pancreatic beta cell with Ca-dependent K.
#[derive(Clone, Debug)]
pub struct ChayNeuron {
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
    pub rho: f64,
    pub alpha_ca: f64,
    pub k_ca: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl ChayNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            n: 0.1,
            ca: 0.1,
            g_ca: 25.0,
            g_k: 1400.0,
            g_kca: 12.0,
            g_l: 7.0,
            e_ca: 100.0,
            e_k: -75.0,
            e_l: -40.0,
            rho: 0.00015,
            alpha_ca: 0.002,
            k_ca: 0.04,
            dt: 0.02,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.dt.is_finite() || self.dt <= 0.0 {
            return 0;
        }

        let v_initial = self.v;
        let mut v = self.v;
        let mut n = self.n;
        let mut ca = self.ca;
        let substeps = (self.dt / 0.001_f64).ceil().max(1.0) as usize;
        let h = self.dt / substeps as f64;
        let mut crossed = false;

        for _ in 0..substeps {
            let m_inf = 1.0 / (1.0 + (-(v + 25.0) / 8.0).clamp(-700.0, 700.0).exp());
            let n_inf = 1.0 / (1.0 + (-(v + 18.0) / 14.0).clamp(-700.0, 700.0).exp());
            let d = (v + 18.0).abs().max(0.01);
            let tau_n = 1.0 / (0.01 * d);
            let ca_denominator = ca + 1.0;
            if ca_denominator <= 0.0 {
                return 0;
            }
            let kca_act = ca / ca_denominator;
            let i_ca = self.g_ca * m_inf * (v - self.e_ca);
            let i_k = self.g_k * n * (v - self.e_k);
            let i_kca = self.g_kca * kca_act * (v - self.e_k);
            let i_l = self.g_l * (v - self.e_l);

            let v_next = v + (-i_ca - i_k - i_kca - i_l + current) * h;
            let n_next = n + (n_inf - n) / tau_n.max(0.01) * h;
            let ca_next = ca + self.rho * (-self.alpha_ca * i_ca - self.k_ca * ca) * h;
            if !v_next.is_finite()
                || !n_next.is_finite()
                || !ca_next.is_finite()
                || !(-200.0..=200.0).contains(&v_next)
                || !(0.0..=1.0).contains(&n_next)
                || !(0.0..=100.0).contains(&ca_next)
            {
                return 0;
            }
            crossed = crossed || (v_next >= self.v_threshold && v < self.v_threshold);
            v = v_next;
            n = n_next;
            ca = ca_next;
        }

        self.v = v;
        self.n = n;
        self.ca = ca;
        if crossed && v_initial < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.n = 0.1;
        self.ca = 0.1;
    }
}
impl Default for ChayNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = ChayNeuron::default();
        let constructed = ChayNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn chay_drive_changes_state_without_leaving_physical_bounds() {
        let mut rest = ChayNeuron::new();
        let mut driven = ChayNeuron::new();
        for _ in 0..500 {
            rest.step(0.0);
            driven.step(5.0);
        }
        assert!(driven.v > rest.v);
        assert!((0.0..=1.0).contains(&driven.n));
        assert!(driven.ca >= 0.0);
    }

    #[test]
    fn chay_reset_clears_state() {
        let mut n = ChayNeuron::new();
        for _ in 0..1000 {
            n.step(20.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn chay_bounded() {
        let mut n = ChayNeuron::new();
        for _ in 0..5000 {
            n.step(200.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn chay_ca_nonneg() {
        let mut n = ChayNeuron::new();
        for _ in 0..5000 {
            n.step(20.0);
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative");
    }

    #[test]
    fn chay_nan_no_panic() {
        ChayNeuron::new().step(f64::NAN);
    }

    #[test]
    fn chay_negative_no_crash() {
        let mut n = ChayNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
}
