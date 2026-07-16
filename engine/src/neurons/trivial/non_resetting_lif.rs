// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Non-resetting LIF Neuron

/// Non-resetting LIF with adaptive threshold. Brette 2004.
#[derive(Clone, Debug)]
pub struct NonResettingLIFNeuron {
    pub v: f64,
    pub theta: f64,
    pub v_rest: f64,
    pub theta_rest: f64,
    pub delta_theta: f64,
    pub tau_m: f64,
    pub tau_theta: f64,
    pub r_m: f64,
    pub dt: f64,
}

impl NonResettingLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            theta: -50.0,
            v_rest: -65.0,
            theta_rest: -50.0,
            delta_theta: 5.0,
            tau_m: 10.0,
            tau_theta: 50.0,
            r_m: 1.0,
            dt: 0.1,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) + self.r_m * current) / self.tau_m * self.dt;
        self.theta += (-(self.theta - self.theta_rest)) / self.tau_theta * self.dt;
        if self.v >= self.theta {
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

impl Default for NonResettingLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_resetting_threshold_increases() {
        let mut n = NonResettingLIFNeuron::new();
        let initial = n.theta;
        for _ in 0..5000 {
            n.step(30.0);
        }
        assert!(n.theta > initial);
    }
    #[test]
    fn nrlif_reset_clears_state() {
        let mut n = NonResettingLIFNeuron::new();
        for _ in 0..500 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.theta - n.theta_rest).abs() < 1e-10);
    }
    #[test]
    fn nrlif_bounded() {
        let mut n = NonResettingLIFNeuron::new();
        for _ in 0..5000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn nrlif_nan_no_panic() {
        NonResettingLIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn nrlif_negative_no_crash() {
        let mut n = NonResettingLIFNeuron::new();
        for _ in 0..500 {
            n.step(-10.0);
        }
        assert!(n.v.is_finite());
    }
}
