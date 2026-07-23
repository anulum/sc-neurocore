// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Akida Neuron Emulator

/// Akida — BrainChip event-domain rank-order neuron.
#[derive(Clone, Debug)]
pub struct AkidaNeuron {
    pub v: i32,
    pub threshold: i32,
    pub modulation: f64,
    pub rank: i32,
    pub spiked: bool,
}

impl AkidaNeuron {
    pub fn new(threshold: i32) -> Self {
        Self {
            v: 0,
            threshold,
            modulation: 0.75,
            rank: 0,
            spiked: false,
        }
    }
    pub fn step(&mut self, weight: f64) -> i32 {
        if self.spiked {
            return 0;
        }
        self.v += (weight * self.modulation.powi(self.rank)) as i32;
        self.rank += 1;
        if self.v >= self.threshold {
            self.spiked = true;
            1
        } else {
            0
        }
    }
    pub fn reset(&mut self) {
        self.v = 0;
        self.rank = 0;
        self.spiked = false;
    }
}
impl Default for AkidaNeuron {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn akida_fires() {
        let mut n = AkidaNeuron::default();
        let t: i32 = (0..10).map(|_| n.step(50.0)).sum();
        assert!(t > 0);
    }
    #[test]
    fn akida_silent() {
        let mut n = AkidaNeuron::default();
        let t: i32 = (0..100).map(|_| n.step(0.0)).sum();
        assert_eq!(t, 0);
    }
    #[test]
    fn akida_reset() {
        let mut n = AkidaNeuron::default();
        for _ in 0..10 {
            n.step(50.0);
        }
        n.reset();
    }
    #[test]
    fn akida_nan_no_panic() {
        AkidaNeuron::default().step(f64::NAN);
    }
}
