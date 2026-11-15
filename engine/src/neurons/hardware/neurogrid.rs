// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — NeuroGrid Neuron Emulator

#![deny(missing_docs)]

/// NeuroGrid — Boahen 2014 subthreshold analog 2-compartment.
#[derive(Clone, Debug)]
pub struct NeuroGridNeuron {
    /// Somatic potential (mV); it carries the exponential spike initiation.
    pub v_s: f64,
    /// Dendritic potential (mV); it receives the input current.
    pub v_d: f64,
    /// Somatic membrane time constant (ms).
    pub tau_s: f64,
    /// Dendritic membrane time constant (ms).
    pub tau_d: f64,
    /// Normalised soma-dendrite coupling conductance (dimensionless).
    pub g_c: f64,
    /// Slope factor of the exponential spike initiation (mV).
    pub delta_t: f64,
    /// Resting potential (mV).
    pub v_rest: f64,
    /// Spike-initiation threshold (mV).
    pub v_threshold: f64,
    /// Spike cutoff (peak) potential (mV).
    pub v_peak: f64,
    /// Reset potential (mV).
    pub v_reset: f64,
    /// RK4 step (ms); each call takes one.
    pub dt: f64,
}

impl NeuroGridNeuron {
    /// Create a neuron with both compartments at -65 mV, the default parameters and
    /// `dt` = 0.1 ms.
    pub fn new() -> Self {
        Self {
            v_s: -65.0,
            v_d: -65.0,
            tau_s: 20.0,
            tau_d: 50.0,
            g_c: 0.5,
            delta_t: 2.0,
            v_rest: -65.0,
            v_threshold: -50.0,
            v_peak: 20.0,
            v_reset: -65.0,
            dt: 0.1,
        }
    }
    fn valid(&self) -> bool {
        self.v_s.is_finite()
            && self.v_d.is_finite()
            && self.tau_s.is_finite()
            && self.tau_s > 0.0
            && self.tau_d.is_finite()
            && self.tau_d > 0.0
            && self.g_c.is_finite()
            && self.g_c >= 0.0
            && self.delta_t.is_finite()
            && self.delta_t > 0.0
            && self.v_rest.is_finite()
            && self.v_threshold.is_finite()
            && self.v_peak.is_finite()
            && self.v_reset.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
    }

    fn derivatives(&self, v_s: f64, v_d: f64, current: f64) -> (f64, f64) {
        let v_s_eff = v_s.min(self.v_peak);
        let dv_d = (-(v_d - self.v_rest) + current - self.g_c * (v_d - v_s_eff)) / self.tau_d;
        let exp_arg = ((v_s_eff - self.v_threshold) / self.delta_t).min(20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv_s = (-(v_s_eff - self.v_rest) + exp_term + self.g_c * (v_d - v_s_eff)) / self.tau_s;
        (dv_s, dv_d)
    }

    fn rk4_substep(&self, v_s: f64, v_d: f64, current: f64) -> (f64, f64) {
        let dt = self.dt;
        let (k1s, k1d) = self.derivatives(v_s, v_d, current);
        let (k2s, k2d) = self.derivatives(v_s + 0.5 * dt * k1s, v_d + 0.5 * dt * k1d, current);
        let (k3s, k3d) = self.derivatives(v_s + 0.5 * dt * k2s, v_d + 0.5 * dt * k2d, current);
        let (k4s, k4d) = self.derivatives(v_s + dt * k3s, v_d + dt * k3d, current);
        (
            v_s + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0,
            v_d + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0,
        )
    }

    /// Advance one classical RK4 step with `current` injected into the dendrite; return 1
    /// when the soma reaches `v_peak`, else 0.
    ///
    /// The dendrite relaxes to `v_rest` and couples to the soma through `g_c`. The soma adds
    /// `delta_t * exp((v_s - v_threshold) / delta_t)`; the exponent is capped at 20, and
    /// `v_s` is capped at `v_peak` inside the equations. A spike sets `v_s` to `v_reset`; the
    /// dendrite takes its new value and is not reset. A non-finite input, an invalid state or
    /// parameter, or a non-finite result returns 0 and leaves the state unchanged.
    pub fn step(&mut self, current: f64) -> i32 {
        if !current.is_finite() || !self.valid() {
            return 0;
        }
        let (next_v_s, next_v_d) = self.rk4_substep(self.v_s, self.v_d, current);
        if !next_v_s.is_finite() || !next_v_d.is_finite() {
            return 0;
        }
        self.v_d = next_v_d;
        if next_v_s >= self.v_peak {
            self.v_s = self.v_reset;
            1
        } else {
            self.v_s = next_v_s;
            0
        }
    }
    /// Set both compartments to -65 mV (not `v_rest` or `v_reset`); the parameters are kept.
    pub fn reset(&mut self) {
        self.v_s = -65.0;
        self.v_d = -65.0;
    }
}
impl Default for NeuroGridNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "neurogrid_tests.rs"]
mod tests;
