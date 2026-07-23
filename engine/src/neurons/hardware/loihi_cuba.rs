// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Loihi CUBA Neuron Emulator

/// Loihi CUBA LIF — Intel Loihi 1 fixed-point neuron. Davies et al. 2018.
#[derive(Clone, Debug)]
pub struct LoihiCUBANeuron {
    pub v: i32,
    pub u: i32,
    pub tau_v: i32,
    pub tau_u: i32,
    pub v_threshold: i32,
    pub v_reset: i32,
}

impl LoihiCUBANeuron {
    pub fn new() -> Self {
        Self {
            v: 0,
            u: 0,
            tau_v: 10,
            tau_u: 5,
            v_threshold: 1000,
            v_reset: 0,
        }
    }
    pub fn step(&mut self, weighted_input: i32) -> i32 {
        self.u = self.u - self.u / self.tau_u + weighted_input;
        self.v = self.v - self.v / self.tau_v + self.u;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0;
        self.u = 0;
    }
}
impl Default for LoihiCUBANeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loihi_cuba_fires() {
        let mut n = LoihiCUBANeuron::new();
        let t: i32 = (0..200).map(|_| n.step(100)).sum();
        assert!(t > 0);
    }
    #[test]
    fn loihi_cuba_silent() {
        let mut n = LoihiCUBANeuron::new();
        let t: i32 = (0..200).map(|_| n.step(0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn loihi_cuba_reset() {
        let mut n = LoihiCUBANeuron::new();
        for _ in 0..50 {
            n.step(100);
        }
        n.reset();
        assert_eq!(n.v, 0);
        assert_eq!(n.u, 0);
    }
    #[test]
    fn loihi_cuba_bounded() {
        let mut n = LoihiCUBANeuron::new();
        for _ in 0..1000 {
            n.step(10000);
        }
    }
}
