// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Exponential integrate-and-fire neuron

/// Exponential IF (no adaptation). Fourcaud-Trocmé et al. 2003.
#[derive(Clone, Debug)]
pub struct ExpIfNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub v_rh: f64,
    pub delta_t: f64,
    pub tau: f64,
    pub dt: f64,
    pub refractory_period: f64,
    pub refractory_remaining: f64,
    /// False preserves the historical SC RK4 recurrence; true selects source RK2.
    pub source_profile: bool,
    pub inv_delta_t: f64,
    pub dt_div_tau: f64,
}

/// Explicit rejection classes for checked ExpIF execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpIfError {
    InvalidInput,
    InvalidState,
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

    /// Preserve the historical scalar ABI while failing closed on rejection.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Advance one checked update without mutating the receiver on rejection.
    pub fn try_step(&mut self, current: f64) -> Result<i32, ExpIfError> {
        if !self.v.is_finite()
            || !current.is_finite()
            || !self.v_rest.is_finite()
            || !self.v_reset.is_finite()
            || !self.v_threshold.is_finite()
            || !self.v_rh.is_finite()
            || !self.delta_t.is_finite()
            || !self.tau.is_finite()
            || !self.dt.is_finite()
            || !self.refractory_period.is_finite()
            || !self.refractory_remaining.is_finite()
            || self.delta_t <= 0.0
            || self.tau <= 0.0
            || self.dt <= 0.0
            || self.refractory_period < 0.0
            || self.refractory_remaining < 0.0
            || self.refractory_remaining > self.refractory_period
            || self.v_threshold <= self.v_rh
            || self.v >= self.v_threshold
            || self.v_rest >= self.v_threshold
            || self.v_reset >= self.v_threshold
            || (self.source_profile
                && (self.v_rest != -65.0
                    || self.v_reset != -68.0
                    || self.v_threshold != -30.0
                    || self.v_rh != -59.9
                    || self.delta_t != 3.48
                    || self.tau != 10.0
                    || self.dt >= 0.02
                    || self.refractory_period != 1.7))
        {
            return Err(if !current.is_finite() {
                ExpIfError::InvalidInput
            } else {
                ExpIfError::InvalidState
            });
        }

        if self.refractory_remaining > 0.0 {
            self.refractory_remaining = (self.refractory_remaining - self.dt).max(0.0);
            self.v = self.v_reset;
            return Ok(0);
        }

        let inv_delta_t = 1.0 / self.delta_t;
        let k1 = self.rhs(self.v, current, inv_delta_t);
        let predictor = self.v + self.dt * k1;
        let k2 = self.rhs(
            if self.source_profile {
                predictor
            } else {
                self.v + 0.5 * self.dt * k1
            },
            current,
            inv_delta_t,
        );
        let (k3, k4, next_v) = if self.source_profile {
            (0.0, 0.0, self.v + 0.5 * self.dt * (k1 + k2))
        } else {
            let k3 = self.rhs(self.v + 0.5 * self.dt * k2, current, inv_delta_t);
            let k4 = self.rhs(self.v + self.dt * k3, current, inv_delta_t);
            (
                k3,
                k4,
                self.v + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            )
        };
        if !k1.is_finite()
            || !k2.is_finite()
            || !k3.is_finite()
            || !k4.is_finite()
            || !next_v.is_finite()
        {
            return Err(ExpIfError::NonFiniteUpdate);
        }

        self.inv_delta_t = inv_delta_t;
        self.dt_div_tau = self.dt / self.tau;
        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_remaining = self.refractory_period;
            Ok(1)
        } else {
            self.v = next_v;
            Ok(0)
        }
    }

    /// Run a checked, failure-atomic batch and return aligned state/event rows.
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
