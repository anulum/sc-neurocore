// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SpiNNaker 2 Neuron Emulator

#![deny(missing_docs)]

/// SpiNNaker2 — TU Dresden ARM Cortex-M4F fixed-point LIF.
#[derive(Clone, Debug)]
pub struct SpiNNaker2Neuron {
    /// Membrane potential (integer fixed-point units); a spike sets it to `v_reset`.
    pub v: i32,
    /// Resting potential (integer fixed-point units) (dimensionless).
    pub v_rest: i32,
    /// Post-spike reset potential (integer fixed-point units) (dimensionless).
    pub v_reset: i32,
    /// Firing threshold (integer fixed-point units) (dimensionless).
    pub v_threshold: i32,
    /// Integer multiplier for the exponential membrane decay (dimensionless).
    pub decay_mult: i32,
    /// Right-shift amount paired with the decay multiplier (dimensionless).
    pub decay_shift: i32,
    /// Refractory period in simulation time-steps (dimensionless).
    pub refrac_steps: i32,
    /// Refractory steps left; a spike sets it to `refrac_steps`.
    pub refrac_count: i32,
}

impl SpiNNaker2Neuron {
    /// Create a neuron at rest with the default fixed-point parameters (decay 243 / 256 per
    /// step, threshold 1024, two refractory steps).
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
    /// Advance one step with the integer input and return 1 on a spike, else 0.
    ///
    /// While `refrac_count` is positive, the step counts it down, leaves `v` unchanged and
    /// returns 0. Otherwise `v` becomes
    /// `((v - v_rest) * decay_mult >> decay_shift) + v_rest + current`; the product wraps,
    /// and the arithmetic shift rounds down. `decay_shift` must be in `0..32`. Reaching
    /// `v_threshold` sets `v` to `v_reset` and starts the refractory count.
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
    /// Return `v` to `v_rest` and clear the refractory count; the parameters are kept.
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
#[path = "spinnaker2_tests.rs"]
mod tests;
