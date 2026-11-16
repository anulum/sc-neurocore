// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Exponential integrate-and-fire neuron

#![deny(missing_docs)]

mod step;

/// Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003.
#[derive(Clone, Debug)]
pub struct ExpIfNeuron {
    /// Membrane potential (mV), below `v_threshold` between steps.
    pub v: f64,
    /// Resting potential (mV).
    pub v_rest: f64,
    /// Potential after a spike (mV).
    pub v_reset: f64,
    /// Spike cutoff (mV); the exponential term is evaluated at most at this voltage.
    pub v_threshold: f64,
    /// Rheobase threshold of the exponential term (mV).
    pub v_rh: f64,
    /// Slope factor of the exponential spike initiation (mV).
    pub delta_t: f64,
    /// Membrane time constant (ms).
    pub tau: f64,
    /// Step (ms).
    pub dt: f64,
    /// Refractory duration after a spike (ms).
    pub refractory_period: f64,
    /// Refractory time left (ms).
    pub refractory_remaining: f64,
    /// False preserves the historical SC RK4 recurrence; true selects source RK2.
    pub source_profile: bool,
    /// Cached `1 / delta_t`, written by each accepted integration step (a refractory step
    /// leaves it); no step reads it.
    pub inv_delta_t: f64,
    /// Cached `dt / tau`, written by each accepted integration step (a refractory step
    /// leaves it); no step reads it.
    pub dt_div_tau: f64,
}

/// Explicit rejection classes for checked ExpIF execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpIfError {
    /// The input current is non-finite.
    InvalidInput,
    /// A state or parameter is non-finite or out of range, or, in the source profile, a
    /// fixed parameter was changed or `dt` is not below 0.02 ms.
    InvalidState,
    /// An integration stage or the candidate voltage is non-finite.
    NonFiniteUpdate,
}

/// Aligned voltage, refractory-state, and event traces from one complete batch.
pub type ExpIfCompleteTrace = (Vec<f64>, Vec<f64>, Vec<u8>);

impl Default for ExpIfNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpIfNeuron {
    /// Create the historical SC profile: RK4, spike cutoff 30 mV, `dt` = 0.02 ms and no
    /// refractory period.
    pub fn new() -> Self {
        Self {
            v: -65.0,
            v_rest: -65.0,
            v_reset: -68.0,
            v_threshold: 30.0,
            v_rh: -59.9,
            delta_t: 3.48,
            tau: 10.0,
            dt: 0.02,
            refractory_period: 0.0,
            refractory_remaining: 0.0,
            source_profile: false,
            inv_delta_t: 1.0 / 3.48,
            dt_div_tau: 0.02 / 10.0,
        }
    }

    /// Construct the fitted Fourcaud-Trocmé protocol's deterministic RK2 lane.
    ///
    /// Spike cutoff -30 mV, `dt` = 0.01 ms and a 1.7 ms refractory period. `try_step`
    /// refuses a change to the profile's fixed parameters and any `dt` of 0.02 ms or more.
    pub fn fourcaud_trocme_2003() -> Self {
        Self {
            v_threshold: -30.0,
            dt: 0.01,
            refractory_period: 1.7,
            source_profile: true,
            dt_div_tau: 0.01 / 10.0,
            ..Self::new()
        }
    }

    /// Preserve the historical scalar ABI while failing closed on rejection: a refused step
    /// returns 0 and leaves the state unchanged.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run a checked, failure-atomic batch and return aligned state/event rows.
    ///
    /// Each row holds the post-step `v`, refractory time left and event. The state is
    /// committed only when every step succeeds; an error leaves it unchanged.
    pub fn simulate_complete(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<ExpIfCompleteTrace, ExpIfError> {
        let mut candidate = self.clone();
        let mut voltage = Vec::with_capacity(n_steps);
        let mut refractory = Vec::with_capacity(n_steps);
        let mut events = Vec::with_capacity(n_steps);
        for _ in 0..n_steps {
            let event = candidate.try_step(current)?;
            voltage.push(candidate.v);
            refractory.push(candidate.refractory_remaining);
            events.push(event as u8);
        }
        *self = candidate;
        Ok((voltage, refractory, events))
    }

    /// Return `v` to `v_rest` and clear the refractory time; the parameters and the profile
    /// are kept.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_remaining = 0.0;
    }

    fn rhs(&self, v: f64, current: f64, inv_delta_t: f64) -> f64 {
        if !v.is_finite() {
            return f64::NAN;
        }
        let bounded_v = v.min(self.v_threshold);
        let exp_arg = (bounded_v - self.v_rh) * inv_delta_t;
        let exp_term = self.delta_t * exp_arg.exp();
        (-(bounded_v - self.v_rest) + exp_term + current) / self.tau
    }
}

#[cfg(test)]
#[path = "exp_if_tests.rs"]
mod tests;
