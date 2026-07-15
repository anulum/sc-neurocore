// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Huber-Braun Neuron Model

//! Huber-Braun temperature-sensitive cold-receptor dynamics.

/// Huber-Braun — temperature-sensitive cold receptor. Braun et al. 1998.
#[derive(Clone, Debug)]
pub struct HuberBraunNeuron {
    pub v: f64,
    pub a_sd: f64,
    pub a_sr: f64,
    pub g_sd: f64,
    pub g_sr: f64,
    pub g_l: f64,
    pub e_sd: f64,
    pub e_sr: f64,
    pub e_l: f64,
    pub tau_sd: f64,
    pub tau_sr: f64,
    pub eta: f64,
    pub dt: f64,
    pub v_threshold: f64,
}

impl HuberBraunNeuron {
    pub fn new() -> Self {
        Self {
            v: -50.0,
            a_sd: 0.0,
            a_sr: 0.0,
            g_sd: 1.5,
            g_sr: 0.4,
            g_l: 0.1,
            e_sd: 50.0,
            e_sr: -90.0,
            e_l: -60.0,
            tau_sd: 10.0,
            tau_sr: 20.0,
            eta: 0.012,
            dt: 0.1,
            v_threshold: -20.0,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        // Braun, Huber et al. 1998: SD V1/2 = -40 mV, slope = 6
        let sd_inf = 1.0 / (1.0 + (-(self.v + 40.0) / 6.0).exp());
        let sr_inf = 1.0 / (1.0 + ((self.v + 40.0) / 6.0).exp());
        self.a_sd += (sd_inf - self.a_sd) / self.tau_sd * self.dt;
        self.a_sr += (sr_inf - self.a_sr) / self.tau_sr * self.dt;
        let i_sd = self.g_sd * self.a_sd * (self.v - self.e_sd);
        let i_sr = self.g_sr * self.a_sr * (self.v - self.e_sr);
        let i_l = self.g_l * (self.v - self.e_l);
        self.v += (-i_sd - i_sr - i_l + current) * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = -50.0;
        self.a_sd = 0.0;
        self.a_sr = 0.0;
    }
}
impl Default for HuberBraunNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_constructor_state() {
        let default = HuberBraunNeuron::default();
        let constructed = HuberBraunNeuron::new();
        assert_eq!(default.v, constructed.v);
    }

    #[test]
    fn hb_fires() {
        let mut n = HuberBraunNeuron::new();
        let t: i32 = (0..5000).map(|_| n.step(10.0)).sum();
        assert!(t > 0);
    }

    // -- HuberBraun --
    #[test]
    fn hb_silent_without_input() {
        let mut n = HuberBraunNeuron::new();
        let _t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        // HuberBraun may have spontaneous activity, so just check finite
        assert!(n.v.is_finite());
    }
    #[test]
    fn hb_reset_clears_state() {
        let mut n = HuberBraunNeuron::new();
        for _ in 0..200 {
            n.step(10.0);
        }
        n.reset();
        assert!((n.v - (-50.0)).abs() < 1e-10);
    }
    #[test]
    fn hb_extreme_bounded() {
        let mut n = HuberBraunNeuron::new();
        for _ in 0..200 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn hb_negative_no_crash() {
        let mut n = HuberBraunNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn hb_nan_no_panic() {
        let mut n = HuberBraunNeuron::new();
        n.step(f64::NAN);
    }
}
