// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — BrainScaleS AdEx Neuron Emulator

#![deny(missing_docs)]

/// BrainScaleS AdEx — Heidelberg analog wafer-scale. Schemmel et al. 2010.
#[derive(Clone, Debug)]
pub struct BrainScaleSAdExNeuron {
    /// Membrane potential (mV).
    pub v: f64,
    /// Adaptation, in the units of the input; it is subtracted from the input in the
    /// membrane equation.
    pub w: f64,
    /// Resting / leak reversal potential (mV).
    pub v_rest: f64,
    /// Membrane potential reset to after a spike (mV).
    pub v_reset: f64,
    /// Voltage cutoff at which a spike is registered (mV).
    pub v_threshold: f64,
    /// Slope factor of the exponential spike initiation (mV).
    pub delta_t: f64,
    /// Rheobase threshold of the exponential term (mV).
    pub v_rh: f64,
    /// Membrane time constant (biological-time equivalent) (ms).
    pub tau: f64,
    /// Adaptation time constant (biological-time equivalent) (ms).
    pub tau_w: f64,
    /// Subthreshold adaptation conductance (nS).
    pub a: f64,
    /// Spike-triggered adaptation current increment (pA).
    pub b: f64,
    /// Hardware acceleration factor relative to biological time (dimensionless). The step does
    /// not read it; it labels the emulated hardware.
    pub hw_speedup: f64,
    /// Euler step in biological time (ms).
    pub dt: f64,
}

impl BrainScaleSAdExNeuron {
    /// Create a neuron at rest (-65 mV, no adaptation) with the default parameters and
    /// `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            w: 0.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: -50.0,
            delta_t: 2.0,
            v_rh: -55.0,
            tau: 20.0,
            tau_w: 100.0,
            a: 0.5,
            b: 7.0,
            hw_speedup: 1000.0,
            dt: 0.1,
        }
    }
    /// Advance one forward-Euler step and return 1 on a spike, else 0.
    ///
    /// `tau dv/dt = -(v - v_rest) + delta_t * exp((v - v_rh) / delta_t) - w + current`, with
    /// the exponent clamped to `[-20, 20]`, and `tau_w dw/dt = a * (v - v_rest) - w`; both
    /// use the old state. Reaching `v_threshold` sets `v` to `v_reset` and adds `b` to `w`.
    /// The input is not validated.
    pub fn step(&mut self, current: f64) -> i32 {
        let exp_arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = (-(self.v - self.v_rest) + exp_term - self.w + current) / self.tau * self.dt;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        self.v += dv;
        self.w += dw;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            self.w += self.b;
            1
        } else {
            0
        }
    }
    /// Return `v` to `v_rest` and clear `w`; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}
impl Default for BrainScaleSAdExNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "brainscales_adex_tests.rs"]
mod tests;
