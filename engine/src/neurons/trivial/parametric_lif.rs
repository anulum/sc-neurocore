// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Parametric LIF Neuron

/// Parametric LIF — learnable decay via sigmoid(a). Fang et al. 2021.
#[derive(Clone, Debug)]
pub struct ParametricLIFNeuron {
    pub v: f64,
    pub a: f64,
    pub threshold: f64,
}

impl ParametricLIFNeuron {
    pub fn new(a: f64, threshold: f64) -> Self {
        Self {
            v: 0.0,
            a,
            threshold,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        let alpha = 1.0 / (1.0 + (-self.a).exp());
        let spike = if self.v >= self.threshold { 1 } else { 0 };
        self.v = alpha * self.v * (1.0 - spike as f64) + current;
        spike
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for ParametricLIFNeuron {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plif_fires() {
        let mut n = ParametricLIFNeuron::default();
        let total: i32 = (0..20).map(|_| n.step(1.5)).sum();
        assert!(total > 0);
    }
    #[test]
    fn plif_silent_without_input() {
        let mut n = ParametricLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn plif_reset_clears_state() {
        let mut n = ParametricLIFNeuron::default();
        for _ in 0..20 {
            n.step(1.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn plif_bounded() {
        let mut n = ParametricLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn plif_nan_no_panic() {
        ParametricLIFNeuron::default().step(f64::NAN);
    }
}
