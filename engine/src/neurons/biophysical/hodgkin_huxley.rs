// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Hodgkin-Huxley Neuron Model

//! Hodgkin-Huxley 1952 conductance-based neuron dynamics.

use super::safe_rate;

/// Hodgkin-Huxley 1952 — 4-ODE ion channel model.
#[derive(Clone, Debug)]
pub struct HodgkinHuxleyNeuron {
    pub v: f64,
    pub m: f64,
    pub h: f64,
    pub n: f64,
    pub c_m: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HodgkinHuxleyNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            m: 0.05,
            h: 0.6,
            n: 0.32,
            c_m: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.4,
            dt: 0.01,
            v_threshold: 0.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let steps = (1.0 / self.dt) as usize;
        for _ in 0..steps {
            let am = safe_rate(0.1, 40.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 65.0) / 18.0).exp();
            let ah = 0.07 * (-(self.v + 65.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 35.0) / 10.0).exp());
            let an = safe_rate(0.01, 55.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 65.0) / 80.0).exp();
            self.m += (am * (1.0 - self.m) - bm * self.m) * self.dt;
            self.h += (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += (an * (1.0 - self.n) - bn * self.n) * self.dt;
            let i_na = self.g_na * self.m.powi(3) * self.h * (self.v - self.e_na);
            let i_k = self.g_k * self.n.powi(4) * (self.v - self.e_k);
            let i_l = self.g_l * (self.v - self.e_l);
            self.v += (-i_na - i_k - i_l + current) / self.c_m * self.dt;
        }
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -65.0;
        self.m = 0.05;
        self.h = 0.6;
        self.n = 0.32;
    }
}
impl Default for HodgkinHuxleyNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = HodgkinHuxleyNeuron::default();
        let constructed = HodgkinHuxleyNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn hh_fires() {
        let mut n = HodgkinHuxleyNeuron::new();
        let t: i32 = (0..100).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }

    // -- HodgkinHuxley --
    #[test]
    fn hh_silent_without_input() {
        let mut n = HodgkinHuxleyNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn hh_reset_clears_state() {
        let mut n = HodgkinHuxleyNeuron::new();
        for _ in 0..100 {
            n.step(10.0);
        }
        n.reset();
        assert!((n.v - (-65.0)).abs() < 1e-10);
        assert!((n.m - 0.05).abs() < 1e-10);
        assert!((n.h - 0.6).abs() < 1e-10);
        assert!((n.n - 0.32).abs() < 1e-10);
    }
    #[test]
    fn hh_extreme_input_bounded() {
        let mut n = HodgkinHuxleyNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn hh_gates_bounded() {
        let mut n = HodgkinHuxleyNeuron::new();
        for _ in 0..500 {
            n.step(10.0);
        }
        assert!(n.m >= 0.0 && n.m <= 1.0, "m={}", n.m);
        assert!(n.h >= 0.0 && n.h <= 1.0, "h={}", n.h);
        assert!(n.n >= 0.0 && n.n <= 1.0, "n={}", n.n);
    }
    #[test]
    fn hh_negative_input_no_crash() {
        let mut n = HodgkinHuxleyNeuron::new();
        for _ in 0..200 {
            n.step(-20.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn hh_nan_input_no_panic() {
        let mut n = HodgkinHuxleyNeuron::new();
        n.step(f64::NAN);
    }
    #[test]
    fn hh_sodium_potassium_opposition() {
        // Na activation drives depolarisation, K drives repolarisation
        let mut n = HodgkinHuxleyNeuron::new();
        for _ in 0..50 {
            n.step(10.0);
        }
        // After spiking, n (K activation) should have risen
        assert!(n.n > 0.32, "K activation n should increase during spiking");
    }
}
