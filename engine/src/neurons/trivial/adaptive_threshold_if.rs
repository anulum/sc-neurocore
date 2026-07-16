// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive-threshold IF Neuron

/// Adaptive-threshold IF. Platkiewicz & Brette 2010.
#[derive(Clone, Debug)]
pub struct AdaptiveThresholdIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub theta_rest: f64,
    pub delta_theta: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub dt: f64,
}

impl AdaptiveThresholdIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            v_rest: -65.0,
            v_reset: -65.0,
            theta_rest: -50.0,
            delta_theta: 5.0,
            tau_m: 10.0,
            tau_theta: 50.0,
            dt: 0.1,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + current) / self.tau_m * self.dt;
        self.theta += (-(self.theta - self.theta_rest)) / self.tau_theta * self.dt;
        if self.v >= self.theta {
            self.v = self.v_reset;
            self.theta += self.delta_theta;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.theta = self.theta_rest;
    }
}

impl Default for AdaptiveThresholdIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_threshold_fires() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        let total: i32 = (0..500).map(|_| n.step(30.0)).sum();
        assert!(total > 0);
    }
    #[test]
    fn atif_silent_without_input() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        let t: i32 = (0..500).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn atif_reset_clears_state() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn atif_bounded() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn atif_nan_no_panic() {
        AdaptiveThresholdIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn atif_threshold_increases_with_spikes() {
        let mut n = AdaptiveThresholdIFNeuron::new();
        let theta_init = n.theta;
        for _ in 0..500 {
            n.step(30.0);
        }
        assert!(n.theta > theta_init, "threshold should adapt after spikes");
    }
}
