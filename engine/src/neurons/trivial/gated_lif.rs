// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Gated LIF Neuron

/// Gated LIF — learnable decay and input gates.
#[derive(Clone, Debug)]
pub struct GatedLIFNeuron {
    pub v: f64,
    pub gate_v: f64,
    pub gate_i: f64,
    pub v_threshold: f64,
    pub dt: f64,
}

impl GatedLIFNeuron {
    pub fn new(gate_v: f64, gate_i: f64, v_threshold: f64) -> Self {
        Self {
            v: 0.0,
            gate_v,
            gate_i,
            v_threshold,
            dt: 1.0,
        }
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.v = self.gate_v * self.v + self.gate_i * current;
        if self.v >= self.v_threshold {
            self.v -= self.v_threshold;
            1
        } else {
            0
        }
    }

    pub fn reset(&mut self) {
        self.v = 0.0;
    }
}

impl Default for GatedLIFNeuron {
    fn default() -> Self {
        Self::new(0.9, 1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gated_lif_fires() {
        let mut n = GatedLIFNeuron::default();
        let total: i32 = (0..20).map(|_| n.step(0.5)).sum();
        assert!(total > 0);
    }
    #[test]
    fn gated_lif_silent_without_input() {
        let mut n = GatedLIFNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn gated_lif_reset_clears_state() {
        let mut n = GatedLIFNeuron::default();
        for _ in 0..20 {
            n.step(0.5);
        }
        n.reset();
        assert!((n.v - 0.0).abs() < 1e-10);
    }
    #[test]
    fn gated_lif_bounded() {
        let mut n = GatedLIFNeuron::default();
        for _ in 0..1000 {
            n.step(100.0);
        }
        assert!(n.v.is_finite());
    }
    #[test]
    fn gated_lif_nan_no_panic() {
        GatedLIFNeuron::default().step(f64::NAN);
    }
}
