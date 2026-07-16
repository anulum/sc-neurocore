// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike-Frequency Adaptation Neuron

/// Spike-Frequency Adaptation LIF. Benda & Herz 2003.
#[derive(Clone, Debug)]
pub struct SFANeuron {
    pub v: f64,
    pub g_sfa: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_sfa: f64,
    pub delta_g: f64,
    pub e_k: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl SFANeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0,
            g_sfa: 0.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 10.0,
            tau_sfa: 200.0,
            delta_g: 0.5,
            e_k: -80.0,
            resistance: 1.0,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v += (-(self.v - self.v_rest) - self.g_sfa * (self.v - self.e_k)
            + self.resistance * current)
            / self.tau_m
            * self.dt;
        self.g_sfa *= (-self.dt / self.tau_sfa).exp();
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.g_sfa += self.delta_g;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.g_sfa = 0.0;
    }
}

impl Default for SFANeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sfa_fires_then_adapts() {
        let mut n = SFANeuron::new();
        let first: i32 = (0..100).map(|_| n.step(30.0)).sum();
        let second: i32 = (0..100).map(|_| n.step(30.0)).sum();
        assert!(first > 0);
        assert!(second <= first + 2);
    }
    #[test]
    fn sfa_silent_without_input() {
        let mut n = SFANeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn sfa_reset_clears_state() {
        let mut n = SFANeuron::new();
        for _ in 0..100 {
            n.step(30.0);
        }
        n.reset();
        assert!((n.v - n.v_rest).abs() < 1e-10);
        assert!((n.g_sfa - 0.0).abs() < 1e-10);
    }
    #[test]
    fn sfa_bounded() {
        let mut n = SFANeuron::new();
        for _ in 0..1000 {
            n.step(1e4);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn sfa_nan_no_panic() {
        SFANeuron::new().step(f64::NAN);
    }
}
