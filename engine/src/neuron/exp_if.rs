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
mod tests {
    use super::ExpIfNeuron;

    fn rk4_reference(neuron: &ExpIfNeuron, current: f64) -> f64 {
        let rhs = |v: f64| {
            let bounded_v = v.min(neuron.v_threshold);
            let exp_arg = (bounded_v - neuron.v_rh) / neuron.delta_t;
            (-(bounded_v - neuron.v_rest) + neuron.delta_t * exp_arg.exp() + current) / neuron.tau
        };
        let k1 = rhs(neuron.v);
        let k2 = rhs(neuron.v + 0.5 * neuron.dt * k1);
        let k3 = rhs(neuron.v + 0.5 * neuron.dt * k2);
        let k4 = rhs(neuron.v + neuron.dt * k3);
        neuron.v + (neuron.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    }

    #[test]
    fn optimised_step_matches_rk4_reference() {
        let mut neuron = ExpIfNeuron::new();
        neuron.v = -60.0;
        let expected = rk4_reference(&neuron, 10.0);
        assert_eq!(neuron.step(10.0), 0);
        assert!((neuron.v - expected).abs() < 1e-12);
    }

    #[test]
    fn strong_input_produces_spikes() {
        let mut neuron = ExpIfNeuron::new();
        let spikes: i32 = (0..2_000).map(|_| neuron.step(500.0)).sum();
        assert!(spikes > 0);
    }

    #[test]
    fn zero_input_remains_silent() {
        let mut neuron = ExpIfNeuron::new();
        let spikes: i32 = (0..500).map(|_| neuron.step(0.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn negative_input_remains_silent() {
        let mut neuron = ExpIfNeuron::new();
        let spikes: i32 = (0..500).map(|_| neuron.step(-100.0)).sum();
        assert_eq!(spikes, 0);
    }

    #[test]
    fn configurable_step_matches_rk4_reference() {
        let mut neuron = ExpIfNeuron::new();
        neuron.v = -60.0;
        neuron.dt = 0.25;
        neuron.tau = 20.0;
        let expected = rk4_reference(&neuron, 12.0);
        assert_eq!(neuron.step(12.0), 0);
        assert!((neuron.v - expected).abs() < 1e-12);
    }

    #[test]
    fn reset_matches_fresh_neuron() {
        let mut neuron = ExpIfNeuron::new();
        for _ in 0..200 {
            neuron.step(500.0);
        }
        neuron.reset();
        let mut fresh = ExpIfNeuron::new();
        let reset_spikes: i32 = (0..100).map(|_| neuron.step(500.0)).sum();
        let fresh_spikes: i32 = (0..100).map(|_| fresh.step(500.0)).sum();
        assert_eq!(reset_spikes, fresh_spikes);
    }

    #[test]
    fn high_input_keeps_voltage_finite() {
        let mut neuron = ExpIfNeuron::new();
        for _ in 0..5_000 {
            neuron.step(1_000.0);
        }
        assert!(neuron.v.is_finite());
    }

    #[test]
    fn enrolled_event_counts_are_stable() {
        for (current, expected) in [(0.0, 0), (5.0, 0), (20.0, 2)] {
            let mut neuron = ExpIfNeuron::new();
            let spikes: i32 = (0..1_000).map(|_| neuron.step(current)).sum();
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn refractory_hold_and_invalid_state_fail_closed() {
        let mut neuron = ExpIfNeuron::new();
        neuron.refractory_period = 1.7;
        while neuron.step(50.0) == 0 {}
        assert_eq!(neuron.refractory_remaining, 1.7);
        for _ in 0..10 {
            assert_eq!(neuron.step(50.0), 0);
            assert_eq!(neuron.v, neuron.v_reset);
        }
        assert!((neuron.refractory_remaining - 1.5).abs() < 1.0e-12);

        let voltage = neuron.v;
        neuron.refractory_remaining = 2.0;
        assert_eq!(neuron.step(0.0), 0);
        assert_eq!((neuron.v, neuron.refractory_remaining), (voltage, 2.0));
    }

    #[test]
    fn ten_thousand_steps_complete_within_smoke_limit() {
        let mut neuron = ExpIfNeuron::new();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            neuron.step(500.0);
        }
        assert!(start.elapsed().as_millis() < 50);
    }

    #[test]
    fn source_factory_freezes_the_fitted_protocol_boundary() {
        let mut neuron = ExpIfNeuron::fourcaud_trocme_2003();
        assert_eq!(neuron.v_threshold, -30.0);
        assert_eq!(neuron.dt, 0.01);
        assert_eq!(neuron.refractory_period, 1.7);
        assert!(neuron.source_profile);
        neuron.dt = 0.02;
        assert_eq!(neuron.try_step(20.0), Err(super::ExpIfError::InvalidState));
    }

    #[test]
    fn checked_batch_is_aligned_and_failure_atomic() {
        let mut neuron = ExpIfNeuron::new();
        let (voltage, refractory, events) = neuron.simulate_complete(1_000, 20.0).unwrap();
        assert_eq!(voltage.len(), 1_000);
        assert_eq!(refractory.len(), 1_000);
        assert_eq!(events.len(), 1_000);
        assert_eq!(events.iter().map(|event| i32::from(*event)).sum::<i32>(), 2);
        assert_eq!(neuron.v, voltage[999]);

        let before = (neuron.v, neuron.refractory_remaining);
        assert_eq!(
            neuron.simulate_complete(2, f64::INFINITY),
            Err(super::ExpIfError::InvalidInput)
        );
        assert_eq!((neuron.v, neuron.refractory_remaining), before);
    }
}
