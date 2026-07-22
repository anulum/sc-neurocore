// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Booth-Rinzel neuron model

//! Booth-Rinzel multi-compartment neuron model.

/// Booth-Rinzel — 2-compartment motoneuron with bistability. Booth et al. 1997.
#[derive(Clone, Debug)]
pub struct BoothRinzelNeuron {
    pub vs: f64,
    pub vd: f64,
    pub h: f64,
    pub n: f64,
    pub q: f64,
    pub ca: f64,
    pub p: f64,
    pub gc: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_ca: f64,
    pub g_kca: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_ca: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl BoothRinzelNeuron {
    pub fn new() -> Self {
        Self {
            vs: -65.0,
            vd: -65.0,
            h: 0.9,
            n: 0.0,
            q: 0.0,
            ca: 0.0,
            p: 0.5,
            gc: 0.1,
            g_na: 120.0,
            g_k: 20.0,
            g_ca: 14.0,
            g_kca: 5.0,
            g_l: 0.51,
            e_na: 55.0,
            e_k: -80.0,
            e_ca: 80.0,
            e_l: -60.0,
            dt: 0.025,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.vs;
        for _ in 0..4 {
            let m_inf = 1.0 / (1.0 + (-(self.vs + 35.0) / 7.8).exp());
            let h_inf = 1.0 / (1.0 + ((self.vs + 55.0) / 7.0).exp());
            let n_inf = 1.0 / (1.0 + (-(self.vs + 28.0) / 15.0).exp());
            let s_inf = 1.0 / (1.0 + (-(self.vd + 22.0) / 5.0).exp());
            let q_inf = 1.0 / (1.0 + (-(self.vd + 35.0) / 2.0).exp());
            let tau_h = (30.0
                / (((self.vs + 50.0) / 15.0).exp() + ((-(self.vs + 50.0)) / 16.0).exp() + 1e-12))
                .max(0.01);
            let tau_n = (7.0
                / (((self.vs + 40.0) / 40.0).exp() + ((-(self.vs + 40.0)) / 50.0).exp() + 1e-12))
                .max(0.01);
            self.h = (self.h + (h_inf - self.h) / tau_h * self.dt).clamp(0.0, 1.0);
            self.n = (self.n + (n_inf - self.n) / tau_n * self.dt).clamp(0.0, 1.0);
            self.q = (self.q + (q_inf - self.q) / 400.0 * self.dt).clamp(0.0, 1.0);
            let chi = (self.ca / 250.0).min(1.0);
            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.vs - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.vs - self.e_k);
            let i_ls = self.g_l * (self.vs - self.e_l);
            let i_sd = (self.gc / self.p) * (self.vs - self.vd);
            let i_ca = self.g_ca * s_inf.powi(2) * (self.vd - self.e_ca);
            let i_kca = self.g_kca * chi * (self.vd - self.e_k);
            let i_ld = self.g_l * (self.vd - self.e_l);
            let i_ds = (self.gc / (1.0 - self.p)) * (self.vd - self.vs);
            self.vs = (self.vs + (-i_na - i_k - i_ls - i_sd + current / self.p) * self.dt)
                .clamp(-200.0, 100.0);
            self.vd = (self.vd + (-i_ca - i_kca - i_ld - i_ds) * self.dt).clamp(-200.0, 100.0);
            self.ca = (self.ca + (0.0025 * (-0.009 * i_ca) - 0.18 * self.ca) * self.dt).max(0.0);
        }
        if self.vs >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.vs = -65.0;
        self.vd = -65.0;
        self.h = 0.9;
        self.n = 0.0;
        self.q = 0.0;
        self.ca = 0.0;
    }
}
impl Default for BoothRinzelNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn booth_fires() {
        let mut n = BoothRinzelNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(5.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn booth_reset() {
        let mut n = BoothRinzelNeuron::new();
        for _ in 0..100 {
            n.step(5.0);
        }
        n.reset();
        assert!((n.vs - (-65.0)).abs() < 1e-10);
    }

    #[test]
    fn booth_bounded() {
        let mut n = BoothRinzelNeuron::new();
        for _ in 0..2000 {
            n.step(50.0);
        }
        assert!(n.vs.is_finite());
    }

    #[test]
    fn booth_nan_no_panic() {
        BoothRinzelNeuron::new().step(f64::NAN);
    }
}
