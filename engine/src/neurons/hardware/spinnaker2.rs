// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SpiNNaker 2 Neuron Emulator

/// SpiNNaker2 — TU Dresden ARM Cortex-M4F fixed-point LIF.
#[derive(Clone, Debug)]
pub struct SpiNNaker2Neuron {
    pub v: i32,
    pub v_rest: i32,
    pub v_reset: i32,
    pub v_threshold: i32,
    pub decay_mult: i32,
    pub decay_shift: i32,
    pub refrac_steps: i32,
    pub refrac_count: i32,
}

impl SpiNNaker2Neuron {
    pub fn new() -> Self {
        Self {
            v: 0,
            v_rest: 0,
            v_reset: 0,
            v_threshold: 1024,
            decay_mult: 243,
            decay_shift: 8,
            refrac_steps: 2,
            refrac_count: 0,
        }
    }
    pub fn step(&mut self, current: i32) -> i32 {
        if self.refrac_count > 0 {
            self.refrac_count -= 1;
            return 0;
        }
        self.v = ((self.v - self.v_rest).wrapping_mul(self.decay_mult) >> self.decay_shift)
            + self.v_rest
            + current;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.refrac_steps;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refrac_count = 0;
    }
}
impl Default for SpiNNaker2Neuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinnaker2_fires() {
        let mut n = SpiNNaker2Neuron::new();
        let t: i32 = (0..200).map(|_| n.step(100)).sum();
        assert!(t > 0);
    }
    #[test]
    fn spinnaker2_silent() {
        let mut n = SpiNNaker2Neuron::new();
        let t: i32 = (0..200).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn spinnaker2_reset() {
        let mut n = SpiNNaker2Neuron::new();
        for _ in 0..50 {
            n.step(100);
        }
        n.reset();
    }
    #[test]
    fn spinnaker2_bounded() {
        let mut n = SpiNNaker2Neuron::new();
        for _ in 0..1000 {
            n.step(10000);
        }
    }
}
