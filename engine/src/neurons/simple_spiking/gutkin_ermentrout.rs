// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Gutkin-Ermentrout Neuron Model

//! Gutkin-Ermentrout cortical neuron dynamics.

/// Gutkin-Ermentrout — reduced cortical neuron with Type-I excitability.
#[derive(Clone, Debug)]
pub struct GutkinErmentroutNeuron {
    pub v: f64,
    pub n: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl GutkinErmentroutNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            n: 0.1,
            g_na: 20.0,
            g_k: 10.0,
            g_l: 8.0,
            e_na: 60.0,
            e_k: -90.0,
            e_l: -80.0,
            dt: 0.05,
            v_threshold: -20.0,
        }
    }
    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.n.is_finite()
            && (0.0..=1.0).contains(&self.n)
            && self.g_na.is_finite()
            && self.g_na >= 0.0
            && self.g_k.is_finite()
            && self.g_k >= 0.0
            && self.g_l.is_finite()
            && self.g_l >= 0.0
            && self.e_na.is_finite()
            && self.e_k.is_finite()
            && self.e_l.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.v_threshold.is_finite()
    }

    fn m_inf(v: f64) -> f64 {
        1.0 / (1.0 + (-(v + 20.0) / 15.0).exp())
    }

    fn n_inf(v: f64) -> f64 {
        1.0 / (1.0 + (-(v + 25.0) / 5.0).exp())
    }

    fn rhs(&self, v: f64, n_gate: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && n_gate.is_finite() && current.is_finite()) {
            return None;
        }
        if !(0.0..=1.0).contains(&n_gate) {
            return None;
        }
        let m_inf = Self::m_inf(v);
        let n_inf = Self::n_inf(v);
        if !(m_inf.is_finite() && n_inf.is_finite()) {
            return None;
        }
        let i_na = self.g_na * m_inf * (v - self.e_na);
        let i_k = self.g_k * n_gate * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = -i_na - i_k - i_l + current;
        let dn = n_inf - n_gate;
        if dv.is_finite() && dn.is_finite() {
            Some((dv, dn))
        } else {
            None
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let Some((k1_v, k1_n)) = self.rhs(self.v, self.n, current) else {
            return 0;
        };
        let Some((k2_v, k2_n)) = self.rhs(
            self.v + 0.5 * self.dt * k1_v,
            self.n + 0.5 * self.dt * k1_n,
            current,
        ) else {
            return 0;
        };
        let Some((k3_v, k3_n)) = self.rhs(
            self.v + 0.5 * self.dt * k2_v,
            self.n + 0.5 * self.dt * k2_n,
            current,
        ) else {
            return 0;
        };
        let Some((k4_v, k4_n)) =
            self.rhs(self.v + self.dt * k3_v, self.n + self.dt * k3_n, current)
        else {
            return 0;
        };
        let next_v = self.v + self.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        let next_n = self.n + self.dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0;
        if !(next_v.is_finite() && next_n.is_finite() && (0.0..=1.0).contains(&next_n)) {
            return 0;
        }
        self.v = next_v;
        self.n = next_n;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.n = 0.1;
    }
}
impl Default for GutkinErmentroutNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = GutkinErmentroutNeuron::default();
        let constructed = GutkinErmentroutNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn gutkin_fires() {
        let mut n = GutkinErmentroutNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(15.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn gutkin_reset_clears_state() {
        let mut n = GutkinErmentroutNeuron::new();
        for _ in 0..500 {
            n.step(15.0);
        }
        n.reset();
        assert!((n.v - (-65.0)).abs() < 1e-10);
    }

    #[test]
    fn gutkin_bounded() {
        let mut n = GutkinErmentroutNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn gutkin_nan_no_panic() {
        GutkinErmentroutNeuron::new().step(f64::NAN);
    }

    #[test]
    fn gutkin_matches_rk4_candidate() {
        fn rhs(n: &GutkinErmentroutNeuron, v: f64, n_gate: f64, current: f64) -> (f64, f64) {
            let m_inf = 1.0 / (1.0 + (-(v + 20.0) / 15.0).exp());
            let n_inf = 1.0 / (1.0 + (-(v + 25.0) / 5.0).exp());
            let i_na = n.g_na * m_inf * (v - n.e_na);
            let i_k = n.g_k * n_gate * (v - n.e_k);
            let i_l = n.g_l * (v - n.e_l);
            (-i_na - i_k - i_l + current, n_inf - n_gate)
        }

        let mut n = GutkinErmentroutNeuron::new();
        let current = 5.0;
        let v0 = n.v;
        let n0 = n.n;
        let dt = n.dt;
        let (k1_v, k1_n) = rhs(&n, v0, n0, current);
        let (k2_v, k2_n) = rhs(&n, v0 + 0.5 * dt * k1_v, n0 + 0.5 * dt * k1_n, current);
        let (k3_v, k3_n) = rhs(&n, v0 + 0.5 * dt * k2_v, n0 + 0.5 * dt * k2_n, current);
        let (k4_v, k4_n) = rhs(&n, v0 + dt * k3_v, n0 + dt * k3_n, current);
        let expected_v = v0 + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        let expected_n = n0 + dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0;

        n.step(current);

        assert!((n.v - expected_v).abs() < 1e-12);
        assert!((n.n - expected_n).abs() < 1e-12);
    }

    #[test]
    fn gutkin_invalid_candidate_preserves_state() {
        let mut n = GutkinErmentroutNeuron {
            dt: 100.0,
            ..Default::default()
        };
        let v0 = n.v;
        let n0 = n.n;
        assert_eq!(n.step(1.0e9), 0);
        assert_eq!(n.v, v0);
        assert_eq!(n.n, n0);
    }

    #[test]
    fn gutkin_negative_no_crash() {
        let mut n = GutkinErmentroutNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
}
