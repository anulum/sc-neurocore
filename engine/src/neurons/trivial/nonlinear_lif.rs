// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Nonlinear LIF Neuron

/// Nonlinear LIF — cubic f-I curve with adaptation. Touboul & Brette 2008.
#[derive(Clone, Debug)]
pub struct NonlinearLIFNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_crit: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub a: f64,
    pub b: f64,
    pub tau_w: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl NonlinearLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_crit: -40.0,
            v_threshold: -20.0,
            v_reset: -65.0,
            a: 0.04,
            b: 0.5,
            tau_w: 100.0,
            c_m: 1.0,
            dt: 0.1,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let v_prev = self.v;
        let cubic = self.a * (self.v - self.v_rest) * (self.v - self.v_crit);
        self.v += (cubic - self.w + current) / self.c_m * self.dt;
        self.w += (self.b * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        if self.v >= self.v_threshold && v_prev < self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}

impl Default for NonlinearLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nlif_fires() {
        let mut n = NonlinearLIFNeuron::new();
        let total: i32 = (0..2000).map(|_| n.step(500.0)).sum();
        assert!(total > 0);
    }
    #[test]
    fn nlif_silent_without_input() {
        let mut n = NonlinearLIFNeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn nlif_reset_clears_state() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..100 {
            n.step(500.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
    }
    #[test]
    fn nlif_bounded() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..2000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn nlif_nan_no_panic() {
        NonlinearLIFNeuron::new().step(f64::NAN);
    }
    #[test]
    fn nlif_recovery_evolves() {
        let mut n = NonlinearLIFNeuron::new();
        for _ in 0..2000 {
            n.step(500.0);
        }
        assert!(
            n.w > 0.0,
            "recovery variable w should increase during spiking"
        );
    }
}
