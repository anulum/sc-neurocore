// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Morris-Lecar Neuron Model

//! Morris-Lecar conductance-based neuron dynamics.

/// Morris-Lecar 1981 — barnacle muscle fiber.
#[derive(Clone, Debug)]
pub struct MorrisLecarNeuron {
    pub v: f64,
    pub w: f64,
    pub c_m: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
    pub v4: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl MorrisLecarNeuron {
    pub fn new() -> Self {
        Self {
            v: -60.0,
            w: 0.0,
            c_m: 20.0,
            g_ca: 4.0,
            g_k: 8.0,
            g_l: 2.0,
            e_ca: 120.0,
            e_k: -84.0,
            e_l: -60.0,
            v1: -1.2,
            v2: 18.0,
            v3: 12.0,
            v4: 17.4,
            phi: 1.0 / 15.0,
            dt: 0.1,
            v_threshold: 0.0,
        }
    }
    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.w.is_finite()
            && self.c_m.is_finite()
            && self.g_ca.is_finite()
            && self.g_k.is_finite()
            && self.g_l.is_finite()
            && self.e_ca.is_finite()
            && self.e_k.is_finite()
            && self.e_l.is_finite()
            && self.v1.is_finite()
            && self.v2.is_finite()
            && self.v3.is_finite()
            && self.v4.is_finite()
            && self.phi.is_finite()
            && self.dt.is_finite()
            && self.v_threshold.is_finite()
            && self.c_m > 0.0
            && self.g_ca > 0.0
            && self.g_k > 0.0
            && self.g_l > 0.0
            && self.v2 > 0.0
            && self.v4 > 0.0
            && self.phi > 0.0
            && self.dt > 0.0
            && (0.0..=1.0).contains(&self.w)
    }

    fn m_inf(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - self.v1) / self.v2).tanh())
    }

    fn w_inf(&self, v: f64) -> f64 {
        0.5 * (1.0 + ((v - self.v3) / self.v4).tanh())
    }

    fn lambda(&self, v: f64) -> f64 {
        self.phi * ((v - self.v3) / (2.0 * self.v4)).cosh()
    }

    fn rhs(&self, v: f64, w: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && w.is_finite() && current.is_finite() && (0.0..=1.0).contains(&w)) {
            return None;
        }
        let m_inf = self.m_inf(v);
        let w_inf = self.w_inf(v);
        let lam = self.lambda(v);
        if !(m_inf.is_finite() && w_inf.is_finite() && lam.is_finite()) {
            return None;
        }
        let i_ca = self.g_ca * m_inf * (v - self.e_ca);
        let i_k = self.g_k * w * (v - self.e_k);
        let i_l = self.g_l * (v - self.e_l);
        let dv = (-i_ca - i_k - i_l + current) / self.c_m;
        let dw = lam * (w_inf - w);
        if dv.is_finite() && dw.is_finite() {
            Some((dv, dw))
        } else {
            None
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return 0;
        }
        let v_prev = self.v;
        let Some((k1_v, k1_w)) = self.rhs(self.v, self.w, current) else {
            return 0;
        };
        let Some((k2_v, k2_w)) = self.rhs(
            self.v + 0.5 * self.dt * k1_v,
            self.w + 0.5 * self.dt * k1_w,
            current,
        ) else {
            return 0;
        };
        let Some((k3_v, k3_w)) = self.rhs(
            self.v + 0.5 * self.dt * k2_v,
            self.w + 0.5 * self.dt * k2_w,
            current,
        ) else {
            return 0;
        };
        let Some((k4_v, k4_w)) =
            self.rhs(self.v + self.dt * k3_v, self.w + self.dt * k3_w, current)
        else {
            return 0;
        };
        let next_v = self.v + self.dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0;
        let next_w = self.w + self.dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0;
        if !(next_v.is_finite() && next_w.is_finite() && (0.0..=1.0).contains(&next_w)) {
            return 0;
        }
        self.v = next_v;
        self.w = next_w;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -60.0;
        self.w = 0.0;
    }
}
impl Default for MorrisLecarNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = MorrisLecarNeuron::default();
        let constructed = MorrisLecarNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn morris_lecar_fires() {
        let mut n = MorrisLecarNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(100.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn ml_silent_without_input() {
        let mut n = MorrisLecarNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }

    #[test]
    fn ml_reset_clears_state() {
        let mut n = MorrisLecarNeuron::new();
        for _ in 0..500 {
            n.step(100.0);
        }
        n.reset();
        assert!((n.v - (-60.0)).abs() < 1e-10);
    }

    #[test]
    fn ml_moderate_input_stable() {
        let mut n = MorrisLecarNeuron::new();
        for _ in 0..2000 {
            n.step(200.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn ml_rk4_separates_from_forward_euler() {
        let mut n = MorrisLecarNeuron::new();
        let v0 = n.v;
        let w0 = n.w;
        let current = 50.0;
        let m_inf = 0.5 * (1.0 + ((v0 - n.v1) / n.v2).tanh());
        let w_inf = 0.5 * (1.0 + ((v0 - n.v3) / n.v4).tanh());
        let lam = n.phi * ((v0 - n.v3) / (2.0 * n.v4)).cosh();
        let i_ca = n.g_ca * m_inf * (v0 - n.e_ca);
        let i_k = n.g_k * w0 * (v0 - n.e_k);
        let i_l = n.g_l * (v0 - n.e_l);
        let euler_v = v0 + (-i_ca - i_k - i_l + current) / n.c_m * n.dt;
        let euler_w = w0 + lam * (w_inf - w0) * n.dt;

        assert_eq!(n.step(current), 0);

        assert!((n.v - euler_v).abs() > 1e-6);
        assert!((n.w - euler_w).abs() > 1e-8);
        assert!(n.v.is_finite());
        assert!((0.0..=1.0).contains(&n.w));
    }

    #[test]
    fn ml_nan_no_panic() {
        let mut n = MorrisLecarNeuron::new();
        let before = (n.v, n.w);
        assert_eq!(n.step(f64::NAN), 0);
        assert_eq!((n.v, n.w), before);
    }

    #[test]
    fn ml_overflow_candidate_preserves_state() {
        let mut n = MorrisLecarNeuron {
            v: 1.0e6,
            w: 0.25,
            ..Default::default()
        };
        let before = (n.v, n.w);
        assert_eq!(n.step(0.0), 0);
        assert_eq!((n.v, n.w), before);
    }

    #[test]
    fn ml_negative_no_crash() {
        let mut n = MorrisLecarNeuron::new();
        for _ in 0..500 {
            n.step(-50.0);
        }
        assert!(n.v.is_finite());
    }

    #[test]
    fn ml_k_gating_bounded() {
        let mut n = MorrisLecarNeuron::new();
        for _ in 0..2000 {
            n.step(100.0);
        }
        assert!(n.w >= 0.0 && n.w <= 1.0, "w={}", n.w);
    }
}
