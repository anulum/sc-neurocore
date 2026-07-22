// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Li-Rinzel astrocyte model

/// Li-Rinzel IP3R astrocyte model — Ca²⁺ dynamics.
#[derive(Clone, Debug)]
pub struct AstrocyteModel {
    pub ca: f64,
    pub h: f64,
    pub ip3: f64,
    pub v_er: f64,
    pub k_er: f64,
    pub v_serca: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d5: f64,
    pub c0: f64,
    pub c1: f64,
    pub dt: f64,
}

impl AstrocyteModel {
    pub fn new() -> Self {
        Self {
            ca: 0.05,
            h: 0.8,
            ip3: 0.5,
            v_er: 0.9,
            k_er: 0.15,
            v_serca: 0.4,
            d1: 0.13,
            d2: 1.049,
            d3: 0.9434,
            d5: 0.08234,
            c0: 2.0,
            c1: 0.185,
            dt: 0.01,
        }
    }
    pub fn step(&mut self, current: f64) -> f64 {
        let ca_er = (self.c0 - self.ca) / self.c1;
        let m_inf = self.ip3 / (self.ip3 + self.d1);
        let n_inf = self.ca / (self.ca + self.d5);
        let j_chan = self.v_er * (m_inf * n_inf * self.h).powi(3) * (ca_er - self.ca);
        let j_leak = self.k_er * (ca_er - self.ca);
        let j_pump = self.v_serca * self.ca.powi(2) / (self.ca.powi(2) + self.k_er.powi(2));
        let q2 = self.d2 * (self.ip3 + self.d1) / (self.ip3 + self.d3);
        let h_inf = q2 / (q2 + self.ca);
        let tau_h = 1.0 / (0.2 * (q2 + self.ca));
        self.ca += (j_chan + j_leak - j_pump + current) * self.dt;
        self.ca = self.ca.max(0.0);
        self.h += (h_inf - self.h) / tau_h * self.dt;
        self.ca
    }
    pub fn reset(&mut self) {
        self.ca = 0.05;
        self.h = 0.8;
        self.ip3 = 0.5;
    }
}
impl Default for AstrocyteModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn astrocyte_ca() {
        let mut n = AstrocyteModel::new();
        let mut max_ca = 0.0_f64;
        for _ in 0..5000 {
            let c = n.step(0.1);
            max_ca = max_ca.max(c);
        }
        assert!(max_ca > 0.05);
    }

    #[test]
    fn astrocyte_reset() {
        let mut n = AstrocyteModel::new();
        for _ in 0..1000 {
            n.step(0.1);
        }
        n.reset();
        assert!((n.ca - 0.05).abs() < 1e-10);
    }

    #[test]
    fn astrocyte_nan_no_panic() {
        AstrocyteModel::new().step(f64::NAN);
    }

    #[test]
    fn astrocyte_ca_nonneg() {
        let mut n = AstrocyteModel::new();
        for _ in 0..5000 {
            n.step(0.1);
        }
        assert!(n.ca >= 0.0, "Ca²⁺ must be non-negative");
    }
}
