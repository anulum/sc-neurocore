// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Adaptive exponential integrate-and-fire neuron

mod simulation;

/// Adaptive Exponential IF neuron. Brette & Gerstner 2005.
/// PyO3 wrapper: `pyo3_neurons::PyAdExNeuron`
#[derive(Clone, Debug)]
pub struct AdExNeuron {
    pub v: f64,
    pub w: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub tau_w: f64,
    pub a: f64,
    pub b: f64,
    pub c_m: f64,
    pub dt: f64,
}

impl Default for AdExNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl AdExNeuron {
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

    /// Advance one checked baseline-Euler step without partial mutation.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.v.is_finite()
            || !self.w.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.v_rh.is_finite()
            || !self.delta_t.is_finite()
            || !self.tau.is_finite()
            || !self.tau_w.is_finite()
            || !self.a.is_finite()
            || !self.b.is_finite()
            || !self.c_m.is_finite()
            || !self.dt.is_finite()
            || !current.is_finite()
            || self.delta_t <= 0.0
            || self.tau <= 0.0
            || self.tau_w <= 0.0
            || self.c_m <= 0.0
            || self.dt <= 0.0
        {
            return Err("invalid AdEx state, parameters, timestep, or input");
        }

        let exp_arg = ((self.v - self.v_rh) / self.delta_t).clamp(-20.0, 20.0);
        let exp_term = self.delta_t * exp_arg.exp();
        let dv = ((-(self.v - self.v_rest) + exp_term) / self.tau + (-self.w + current) / self.c_m)
            * self.dt;
        let dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w * self.dt;
        let next_v = self.v + dv;
        let next_w = self.w + dw;
        if !exp_term.is_finite()
            || !dv.is_finite()
            || !dw.is_finite()
            || !next_v.is_finite()
            || !next_w.is_finite()
        {
            return Err("non-finite AdEx integrator candidate");
        }

        if next_v >= self.v_threshold {
            let spike_w = next_w + self.b;
            if !spike_w.is_finite() {
                return Err("non-finite AdEx spike-adaptation candidate");
            }
            self.v = self.v_reset;
            self.w = spike_w;
            Ok(1)
        } else {
            self.v = next_v;
            self.w = next_w;
            Ok(0)
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.w = 0.0;
    }
}

#[cfg(test)]
mod tests;
