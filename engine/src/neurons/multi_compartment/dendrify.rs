// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dendrify neuron model

//! Dendrify multi-compartment neuron model.

/// Dendrify — two-compartment with active dendritic spike (NMDA-like plateau).
#[derive(Clone, Debug)]
pub struct DendrifyNeuron {
    pub v_s: f64,
    pub v_d: f64,
    pub d_active: bool,
    pub d_timer: f64,
    pub tau_s: f64,
    pub tau_d: f64,
    pub g_c: f64,
    pub d_threshold: f64,
    pub d_amplitude: f64,
    pub d_duration: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    pub dt: f64,
}

impl DendrifyNeuron {
    pub fn new() -> Self {
        Self {
            v_s: -65.0,
            v_d: -65.0,
            d_active: false,
            d_timer: 0.0,
            tau_s: 10.0,
            tau_d: 20.0,
            g_c: 0.8,
            d_threshold: -35.0,
            d_amplitude: 30.0,
            d_duration: 10.0,
            v_rest: -65.0,
            v_threshold: -50.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    pub fn step(&mut self, current: f64) -> i32 {
        let d_input = if self.d_active { self.d_amplitude } else { 0.0 };
        self.v_d += (-(self.v_d - self.v_rest) + current + d_input
            - self.g_c * (self.v_d - self.v_s))
            / self.tau_d
            * self.dt;
        self.v_s +=
            (-(self.v_s - self.v_rest) + self.g_c * (self.v_d - self.v_s)) / self.tau_s * self.dt;
        if self.d_active {
            self.d_timer -= self.dt;
            if self.d_timer <= 0.0 {
                self.d_active = false;
            }
        } else if self.v_d >= self.d_threshold {
            self.d_active = true;
            self.d_timer = self.d_duration;
        }
        if self.v_s >= self.v_threshold {
            self.v_s = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v_s = -65.0;
        self.v_d = -65.0;
        self.d_active = false;
        self.d_timer = 0.0;
    }
}
impl Default for DendrifyNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dendrify_fires() {
        let mut n = DendrifyNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(50.0)).sum();
        assert!(t > 0);
    }

    #[test]
    fn dendrify_reset() {
        let mut n = DendrifyNeuron::new();
        for _ in 0..100 {
            n.step(50.0);
        }
        n.reset();
        assert!((n.v_s - (-65.0)).abs() < 1e-10);
    }

    #[test]
    fn dendrify_bounded() {
        let mut n = DendrifyNeuron::new();
        for _ in 0..2000 {
            n.step(200.0);
        }
        assert!(n.v_s.is_finite());
    }

    #[test]
    fn dendrify_nan_no_panic() {
        DendrifyNeuron::new().step(f64::NAN);
    }
}
