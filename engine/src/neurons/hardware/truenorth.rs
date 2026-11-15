// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — TrueNorth Neuron Emulator

#![deny(missing_docs)]

/// TrueNorth — IBM TrueNorth digital crossbar neuron. Merolla et al. 2014.
#[derive(Clone, Debug)]
pub struct TrueNorthNeuron {
    /// Membrane potential (integer); a spike sets it to `v_reset`.
    pub v: i32,
    /// Integer leak subtracted from the membrane each timestep (dimensionless).
    pub leak: i32,
    /// Integer firing threshold (dimensionless).
    pub threshold: i32,
    /// Integer reset value (dimensionless).
    pub v_reset: i32,
}

impl TrueNorthNeuron {
    /// Create a neuron at 0 with the given threshold, no leak and reset value 0.
    pub fn new(threshold: i32) -> Self {
        Self {
            v: 0,
            leak: 0,
            threshold,
            v_reset: 0,
        }
    }
    /// Add `weighted_input - leak` to `v` and return 1 when `v` reaches `threshold`, which
    /// sets `v` to `v_reset`; else 0.
    pub fn step(&mut self, weighted_input: i32) -> i32 {
        self.v += weighted_input - self.leak;
        if self.v >= self.threshold {
            self.v = self.v_reset;
            1
        } else {
            0
        }
    }
    /// Set `v` to 0 (not `v_reset`); the parameters are kept.
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
#[path = "truenorth_tests.rs"]
mod tests;
