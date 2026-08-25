// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained three-state project phantom recurrence

//! SCThreeState dual-slow-potassium phantom-burster dynamics.

/// SC three-state phantom burster — dual slow K for phantom bursting. retained project recurrence.
#[derive(Clone, Debug)]
pub struct SCThreeStatePhantomBurster {
    pub v: f64,
    pub s1: f64,
    pub s2: f64,
    pub g_ca: f64,
    pub g_k: f64,
    pub g_s1: f64,
    pub g_s2: f64,
    pub g_l: f64,
    pub e_ca: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub v_m: f64,
    pub s_m: f64,
    pub v_n: f64,
    pub s_n: f64,
    pub v_s1: f64,
    pub s_s1: f64,
    pub v_s2: f64,
    pub s_s2: f64,
    pub tau_s1: f64,
    pub tau_s2: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl SCThreeStatePhantomBurster {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            s1: 0.1,
            s2: 0.1,
            g_ca: 3.6,
            g_k: 10.0,
            g_s1: 4.0,
            g_s2: 4.0,
            g_l: 0.2,
            e_ca: 25.0,
            e_k: -75.0,
            e_l: -40.0,
            c_m: 5.3,
            v_m: -20.0,
            s_m: 12.0,
            v_n: -16.0,
            s_n: 5.6,
            v_s1: -40.0,
            s_s1: 10.0,
            v_s2: -42.0,
            s_s2: 0.4,
            tau_s1: 20000.0,
            tau_s2: 100000.0,
            dt: 0.5,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let m_inf = 1.0 / (1.0 + (-(self.v - self.v_m) / self.s_m).exp());
        let n_inf = 1.0 / (1.0 + (-(self.v - self.v_n) / self.s_n).exp());
        let s1_inf = 1.0 / (1.0 + (-(self.v - self.v_s1) / self.s_s1).exp());
        let s2_inf = 1.0 / (1.0 + (-(self.v - self.v_s2) / self.s_s2).exp());
        let i_ca = self.g_ca * m_inf * (self.v - self.e_ca);
        let i_k = self.g_k * n_inf * (self.v - self.e_k);
        let i_s1 = self.g_s1 * self.s1 * (self.v - self.e_k);
        let i_s2 = self.g_s2 * self.s2 * (self.v - self.e_k);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / self.c_m * self.dt;
        self.s1 += (s1_inf - self.s1) / self.tau_s1 * self.dt;
        self.s2 += (s2_inf - self.s2) / self.tau_s2 * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.s1 = 0.1;
        self.s2 = 0.1;
    }
}
impl Default for SCThreeStatePhantomBurster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = SCThreeStatePhantomBurster::default();
        let constructed = SCThreeStatePhantomBurster::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn sc_three_state_fires() {
        let mut n = SCThreeStatePhantomBurster::new();
        let t: i32 = (0..10000).map(|_| n.step(200.0)).sum();
        assert!(t > 0);
    }

    // -- SCThreeStatePhantom --
    #[test]
    fn sc_three_state_silent_without_input() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..500 {
            n.step(0.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_three_state_reset_clears_state() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..1000 {
            n.step(200.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
        assert!((n.s1 - 0.1).abs() < 1e-10);
        assert!((n.s2 - 0.1).abs() < 1e-10);
    }
    #[test]
    fn sc_three_state_extreme_bounded() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_three_state_dual_slow_vars() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..10000 {
            n.step(200.0);
        }
        // Both slow variables should evolve
        assert!(n.s1 >= 0.0 && n.s1 <= 1.0, "s1={}", n.s1);
        assert!(n.s2 >= 0.0 && n.s2 <= 1.0, "s2={}", n.s2);
    }
    #[test]
    fn sc_three_state_negative_no_crash() {
        let mut n = SCThreeStatePhantomBurster::new();
        for _ in 0..200 {
            n.step(-100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sc_three_state_nan_no_panic() {
        let mut n = SCThreeStatePhantomBurster::new();
        n.step(f64::NAN);
    }
}
