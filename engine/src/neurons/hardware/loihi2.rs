// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Loihi 2 Neuron Emulator

#![deny(missing_docs)]

/// Loihi 2 — Intel Loihi 2 three-state integer neuron.
#[derive(Clone, Debug)]
pub struct Loihi2Neuron {
    /// State variable 1, the membrane (integer); a spike sets it to `s1_reset`.
    pub s1: i32,
    /// State variable 2, the input current (integer).
    pub s2: i32,
    /// State variable 3, the spike-driven trace (integer); a spike adds `s3_incr`.
    pub s3: i32,
    /// Decay time constant of state variable 1; each step removes s1 / tau1, truncated towards zero (steps).
    pub tau1: i32,
    /// Decay time constant of state variable 2; each step removes s2 / tau2, truncated towards zero (steps).
    pub tau2: i32,
    /// Decay time constant of state variable 3; each step removes s3 / tau3, truncated towards zero (steps).
    pub tau3: i32,
    /// Cross-coupling weight from state 2 into state 1 (dimensionless).
    pub w12: i32,
    /// Cross-coupling weight from state 3 into state 1 (dimensionless).
    pub w13: i32,
    /// Cross-coupling weight from state 3 into state 2 (dimensionless).
    pub w23: i32,
    /// Firing threshold on state variable 1 (integer units).
    pub s1_threshold: i32,
    /// Post-spike reset value of state variable 1 (dimensionless).
    pub s1_reset: i32,
    /// Increment added to state variable 3 per spike (dimensionless).
    pub s3_incr: i32,
}

impl Loihi2Neuron {
    /// Create a neuron with every state at 0 and the default integer parameters.
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
    /// Advance one chip step with the integer input and return 1 on a spike, else 0.
    ///
    /// In order:
    /// - `s3` decays by `s3 / tau3`;
    /// - `s2` decays by `s2 / tau2` and gains the input plus `w23 * s3`;
    /// - `s1` decays by `s1 / tau1` and gains `w12 * s2 + w13 * s3`.
    ///
    /// The divisions truncate towards zero, and a time constant of 0 panics. Reaching
    /// `s1_threshold` sets `s1` to `s1_reset` and adds `s3_incr` to `s3`.
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
    /// Clear the three states; the parameters are kept.
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
#[path = "loihi2_tests.rs"]
mod tests;
