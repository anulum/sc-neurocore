// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Complementary LIF Neuron

/// Complementary LIF — dual-path excitatory/inhibitory.
#[derive(Clone, Debug)]
pub struct ComplementaryLIFNeuron {
    pub v_pos: f64,
    pub v_neg: f64,
    pub alpha: f64,
    pub v_threshold: f64,
}

impl ComplementaryLIFNeuron {
    pub fn new(tau: f64, dt: f64, v_threshold: f64) -> Self {
        Self {
            v_pos: 0.0,
            v_neg: 0.0,
            alpha: (-dt / tau).exp(),
            v_threshold,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let inp_pos = current.max(0.0);
        let inp_neg = (-current).max(0.0);
        self.v_pos = self.alpha * self.v_pos + inp_pos;
        self.v_neg = self.alpha * self.v_neg + inp_neg;
        let diff = self.v_pos - self.v_neg;
        if diff >= self.v_threshold {
            self.v_pos = 0.0;
            self.v_neg = 0.0;
            1
        } else if diff <= -self.v_threshold {
            self.v_pos = 0.0;
            self.v_neg = 0.0;
            -1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v_pos = 0.0;
        self.v_neg = 0.0;
    }
}

impl Default for ComplementaryLIFNeuron {
    fn default() -> Self {
        Self::new(10.0, 1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clif_positive_spike() {
        let mut n = ComplementaryLIFNeuron::default();
        let total: i32 = (0..20).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }
    #[test]
    fn clif_silent_without_input() {
        let mut n = ComplementaryLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn clif_reset_clears_state() {
        let mut n = ComplementaryLIFNeuron::default();
        for _ in 0..20 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v_pos - 0.0).abs() < 1e-10);
        assert!((n.v_neg - 0.0).abs() < 1e-10);
    }
    #[test]
    fn clif_bounded() {
        let mut n = ComplementaryLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v_pos.is_finite());
    }
    #[test]
    fn clif_nan_no_panic() {
        ComplementaryLIFNeuron::default().step(f64::NAN);
    }
}
