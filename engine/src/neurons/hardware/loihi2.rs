// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Loihi 2 Neuron Emulator

/// Loihi 2 — Intel Loihi 2 three-state integer neuron.
#[derive(Clone, Debug)]
pub struct Loihi2Neuron {
    pub s1: i32,
    pub s2: i32,
    pub s3: i32,
    pub tau1: i32,
    pub tau2: i32,
    pub tau3: i32,
    pub w12: i32,
    pub w13: i32,
    pub w23: i32,
    pub s1_threshold: i32,
    pub s1_reset: i32,
    pub s3_incr: i32,
}

impl Loihi2Neuron {
    pub fn new() -> Self {
        Self {
            s1: 0,
            s2: 0,
            s3: 0,
            tau1: 10,
            tau2: 5,
            tau3: 50,
            w12: 1,
            w13: 0,
            w23: 0,
            s1_threshold: 1000,
            s1_reset: 0,
            s3_incr: 10,
        }
    }
    pub fn step(&mut self, weighted_input: i32) -> i32 {
        self.s3 -= self.s3 / self.tau3;
        self.s2 = self.s2 - self.s2 / self.tau2 + weighted_input + self.w23 * self.s3;
        self.s1 = self.s1 - self.s1 / self.tau1 + self.w12 * self.s2 + self.w13 * self.s3;
        if self.s1 >= self.s1_threshold {
            self.s1 = self.s1_reset;
            self.s3 += self.s3_incr;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.s1 = 0;
        self.s2 = 0;
        self.s3 = 0;
    }
}
impl Default for Loihi2Neuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loihi2_fires() {
        let mut n = Loihi2Neuron {
            tau3: 8,
            ..Loihi2Neuron::new()
        };
        let t: i32 = (0..500).map(|_| n.step(200)).sum();
        assert!(t > 0);
    }
    #[test]
    fn loihi2_silent() {
        let mut n = Loihi2Neuron::new();
        let t: i32 = (0..200).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn loihi2_reset() {
        let mut n = Loihi2Neuron::new();
        for _ in 0..50 {
            n.step(200);
        }
        n.reset();
        assert_eq!(n.s1, 0);
    }
    #[test]
    fn loihi2_bounded() {
        let mut n = Loihi2Neuron {
            tau3: 8,
            ..Loihi2Neuron::new()
        };
        for _ in 0..1000 {
            n.step(10000);
        }
    }
}
