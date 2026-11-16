// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive exponential integrate-and-fire neuron

#![deny(missing_docs)]

mod simulation;
mod step;

/// Adaptive Exponential IF neuron. Brette & Gerstner 2005.
/// PyO3 wrapper: `pyo3_neurons::PyAdExNeuron`
#[derive(Clone, Debug)]
pub struct AdExNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Adaptation current (pA).
    pub w: f64,
    /// Resting / leak reversal potential (mV).
    pub v_rest: f64,
    /// Membrane potential the model is reset to after a spike (mV).
    pub v_reset: f64,
    /// Voltage cutoff at which a spike is registered and reset applied (mV).
    pub v_threshold: f64,
    /// Rheobase threshold of the exponential spike-initiation term (mV).
    pub v_rh: f64,
    /// Slope factor (sharpness) of the exponential spike initiation (mV).
    pub delta_t: f64,
    /// Membrane time constant (C / g_leak) (ms).
    pub tau: f64,
    /// Adaptation time constant (ms).
    pub tau_w: f64,
    /// Subthreshold adaptation conductance coupling voltage to the adaptation current (nS).
    pub a: f64,
    /// Spike-triggered adaptation current increment added to w at each spike (pA).
    pub b: f64,
    /// Membrane capacitance (pF).
    pub c_m: f64,
    /// Euler step (ms).
    pub dt: f64,
}

impl Default for AdExNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl AdExNeuron {
    /// Create a neuron at rest (-65 mV, no adaptation) with the default parameters and
    /// `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -50.0,
            v_rh: -55.0,
            delta_t: 2.0,
            tau: 20.0,
            tau_w: 100.0,
            a: 0.5,
            b: 7.0,
            c_m: 200.0,
            dt: 0.1,
        }
    }

    /// Advance one maintained baseline-Euler step.
    ///
    /// This compatibility surface retains the historical zero-event result on
    /// invalid input. New batch and binding code uses [`Self::try_step`] so a
    /// rejected update cannot be mistaken for a valid quiet timestep.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Return `v` to `v_rest` and clear `w`; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}

#[cfg(test)]
mod tests;
