// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — TrueNorth Neuron Emulator

/// TrueNorth — IBM TrueNorth digital crossbar neuron. Merolla et al. 2014.
#[derive(Clone, Debug)]
pub struct TrueNorthNeuron {
    pub v: i32,
    pub leak: i32,
    pub threshold: i32,
    pub v_reset: i32,
}

impl TrueNorthNeuron {
    pub fn new(threshold: i32) -> Self {
        Self {
            v: 0,
            leak: 0,
            threshold,
            v_reset: 0,
        }
    }
    pub fn step(&mut self, weighted_input: i32) -> i32 {
        self.v += weighted_input - self.leak;
        if self.v >= self.threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0;
    }
}
impl Default for TrueNorthNeuron {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truenorth_fires() {
        let mut n = TrueNorthNeuron::default();
        let t: i32 = (0..10).map(|_| n.step(50)).sum();
        assert!(t > 0);
    }
    #[test]
    fn truenorth_silent() {
        let mut n = TrueNorthNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn truenorth_reset() {
        let mut n = TrueNorthNeuron::default();
        for _ in 0..10 {
            n.step(50);
        }
        n.reset();
        assert_eq!(n.v, 0);
    }
}
