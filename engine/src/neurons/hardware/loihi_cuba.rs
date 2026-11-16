// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Loihi CUBA Neuron Emulator

#![deny(missing_docs)]

/// Loihi CUBA LIF — Intel Loihi 1 fixed-point neuron. Davies et al. 2018.
#[derive(Clone, Debug)]
pub struct LoihiCUBANeuron {
    /// Membrane voltage (integer); a spike sets it to `v_reset`.
    pub v: i32,
    /// Synaptic current (integer).
    pub u: i32,
    /// Membrane-voltage decay time constant; each step removes v / tau_v (steps).
    pub tau_v: i32,
    /// Synaptic-current decay time constant; each step removes u / tau_u (steps).
    pub tau_u: i32,
    /// Integer fixed-point firing threshold (dimensionless).
    pub v_threshold: i32,
    /// Integer fixed-point reset value (dimensionless).
    pub v_reset: i32,
}

impl LoihiCUBANeuron {
    /// Create a neuron with both states at 0 and the default integer parameters.
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
    /// Advance one chip step with the integer input and return 1 on a spike, else 0.
    ///
    /// `u` decays by `u / tau_u` and gains the input; then `v` decays by `v / tau_v` and
    /// gains `u`. The divisions truncate towards zero, and a time constant of 0 panics.
    /// Reaching `v_threshold` sets `v` to `v_reset`.
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
    /// Clear both states; the parameters are kept.
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
#[path = "loihi_cuba_tests.rs"]
mod tests;
