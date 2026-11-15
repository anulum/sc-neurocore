// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SpiNNaker LIF Neuron Emulator

#![deny(missing_docs)]

/// SpiNNaker LIF — ARM Cortex-M4 digital LIF with refractory. Furber et al. 2014.
#[derive(Clone, Debug)]
pub struct SpiNNakerLIFNeuron {
    /// Membrane potential (mV); a spike sets it to `v_reset`.
    pub v: f64,
    /// Resting / leak reversal potential (mV).
    pub v_rest: f64,
    /// Membrane potential reset to after a spike (mV).
    pub v_reset: f64,
    /// Voltage at which a spike is registered (mV).
    pub v_threshold: f64,
    /// Membrane time constant (ms).
    pub tau_m: f64,
    /// Constant offset (bias) input current (nA).
    pub i_offset: f64,
    /// Absolute refractory period (ms).
    pub tau_refrac: f64,
    /// Refractory time left (ms); a spike sets it to `tau_refrac`.
    pub refrac_count: f64,
    /// Step (ms).
    pub dt: f64,
}

impl SpiNNakerLIFNeuron {
    /// Create a neuron at rest (-70 mV) with the default parameters and `dt` = 1 ms.
    pub fn new() -> Self {
        Self {
            v: -70.0,
            v_rest: -70.0,
            v_reset: -70.0,
            v_threshold: -50.0,
            tau_m: 20.0,
            i_offset: 0.0,
            tau_refrac: 2.0,
            refrac_count: 0.0,
            dt: 1.0,
        }
    }
    /// Advance one forward-Euler step and return 1 on a spike, else 0.
    ///
    /// While `refrac_count` is positive, the step counts it down by `dt`, leaves `v`
    /// unchanged and returns 0. Otherwise `v` gains
    /// `(-(v - v_rest) + current + i_offset) / tau_m * dt`, and reaching `v_threshold` sets
    /// `v` to `v_reset` and starts the refractory period. The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        if self.refrac_count > 0.0 {
            self.refrac_count -= self.dt;
            return 0;
        }
        self.v += (-(self.v - self.v_rest) + current + self.i_offset) / self.tau_m * self.dt;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.refrac_count = self.tau_refrac;
            1
        } else {
            0
        }
    }
    /// Return `v` to `v_rest` and clear the refractory time; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refrac_count = 0.0;
    }
}
impl Default for SpiNNakerLIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "spinnaker_lif_tests.rs"]
mod tests;
