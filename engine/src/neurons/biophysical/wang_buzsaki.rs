// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Wang-Buzsaki Neuron Model

//! Wang-Buzsaki fast-spiking interneuron dynamics.

use super::safe_rate;

/// Wang-Buzsaki — fast-spiking GABAergic interneuron. Wang & Buzsáki 1996.
#[derive(Clone, Debug)]
pub struct WangBuzsakiNeuron {
    pub v: f64,
    pub h: f64,
    pub n: f64,
    pub g_na: f64,
    pub g_k: f64,
    pub g_l: f64,
    pub e_na: f64,
    pub e_k: f64,
    pub e_l: f64,
    pub c_m: f64,
    pub phi: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl WangBuzsakiNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            h: 0.8,
            n: 0.1,
            g_na: 35.0,
            g_k: 9.0,
            g_l: 0.1,
            e_na: 55.0,
            e_k: -90.0,
            e_l: -65.0,
            c_m: 1.0,
            phi: 5.0,
            dt: 0.01,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let n_sub = (0.5 / self.dt.max(0.001)) as usize;
        for _ in 0..n_sub {
            let am = safe_rate(0.1, 35.0, self.v, 10.0, 1.0);
            let bm = 4.0 * (-(self.v + 60.0) / 18.0).exp();
            let m_inf = am / (am + bm);
            let ah = 0.07 * (-(self.v + 58.0) / 20.0).exp();
            let bh = 1.0 / (1.0 + (-(self.v + 28.0) / 10.0).exp());
            let an = safe_rate(0.01, 34.0, self.v, 10.0, 0.1);
            let bn = 0.125 * (-(self.v + 44.0) / 80.0).exp();
            self.h += self.phi * (ah * (1.0 - self.h) - bh * self.h) * self.dt;
            self.n += self.phi * (an * (1.0 - self.n) - bn * self.n) * self.dt;
            let i_na = self.g_na * m_inf.powi(3) * self.h * (self.v - self.e_na);
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
        self.h = 0.8;
        self.n = 0.1;
    }
}
impl Default for WangBuzsakiNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = WangBuzsakiNeuron::default();
        let constructed = WangBuzsakiNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn wb_fires() {
        let mut n = WangBuzsakiNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(2.0)).sum();
        assert!(t > 0);
    }

    // -- WangBuzsaki --
    #[test]
    fn wb_silent_without_input() {
        let mut n = WangBuzsakiNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn wb_reset_clears_state() {
        let mut n = WangBuzsakiNeuron::new();
        for _ in 0..100 {
            n.step(2.0);
        }
        n.reset();
        assert!((n.v - (-65.0)).abs() < 1e-10);
    }
    #[test]
    fn wb_extreme_bounded() {
        let mut n = WangBuzsakiNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn wb_fast_spiking_high_rate() {
        // WB model is fast-spiking — should achieve high rates
        let mut n = WangBuzsakiNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(5.0)).sum();
        assert!(t > 10, "WB FS should produce many spikes, got {}", t);
    }
    #[test]
    fn wb_gates_bounded() {
        let mut n = WangBuzsakiNeuron::new();
        for _ in 0..500 {
            n.step(2.0);
        }
        assert!(n.h >= 0.0 && n.h <= 1.0);
        assert!(n.n >= 0.0 && n.n <= 1.0);
    }
    #[test]
    fn wb_negative_no_crash() {
        let mut n = WangBuzsakiNeuron::new();
        for _ in 0..200 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn wb_nan_no_panic() {
        let mut n = WangBuzsakiNeuron::new();
        n.step(f64::NAN);
    }
}
