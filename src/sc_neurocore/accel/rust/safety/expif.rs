// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fail-closed Fourcaud-Trocmé ExpIF recurrence

/// Runtime state and parameters for the deterministic ExpIF recurrence.
#[derive(Debug, Clone)]
pub struct ExpIFNeuron {
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
}

/// Explicit rejection classes for invalid ExpIF updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpIFError {
    InvalidInput,
    InvalidState,
    NonFiniteUpdate,
}

impl Default for ExpIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpIFNeuron {
    /// Construct the source-bound deterministic catalogue defaults.
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
        }
    }

    /// Advance one candidate-first RK4 update without partial mutation.
    pub fn step(&mut self, current: f64) -> Result<i32, ExpIFError> {
        if !current.is_finite() {
            return Err(ExpIFError::InvalidInput);
        }
        if !validate_expif(self) {
            return Err(ExpIFError::InvalidState);
        }

        if self.refractory_remaining > 0.0 {
            self.refractory_remaining = (self.refractory_remaining - self.dt).max(0.0);
            self.v = self.v_reset;
            return Ok(0);
        }

        let k1 = self.rhs(self.v, current);
        let k2 = self.rhs(self.v + 0.5 * self.dt * k1, current);
        let k3 = self.rhs(self.v + 0.5 * self.dt * k2, current);
        let k4 = self.rhs(self.v + self.dt * k3, current);
        let next_v = self.v + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        if !k1.is_finite()
            || !k2.is_finite()
            || !k3.is_finite()
            || !k4.is_finite()
            || !next_v.is_finite()
        {
            return Err(ExpIFError::NonFiniteUpdate);
        }

        if next_v >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_remaining = self.refractory_period;
            Ok(1)
        } else {
            self.v = next_v;
            Ok(0)
        }
    }

    /// Restore resting voltage and clear the refractory hold.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_remaining = 0.0;
    }

    fn rhs(&self, v: f64, current: f64) -> f64 {
        if !v.is_finite() {
            return f64::NAN;
        }
        let bounded_v = v.min(self.v_threshold);
        let arg = (bounded_v - self.v_rh) / self.delta_t;
        let exp_term = self.delta_t * arg.exp();
        (-(bounded_v - self.v_rest) + exp_term + current) / self.tau
    }
}

/// Return whether every runtime invariant required by [`ExpIFNeuron::step`] holds.
pub fn validate_expif(state: &ExpIFNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.v_rh.is_finite()
        && state.delta_t.is_finite()
        && state.tau.is_finite()
        && state.dt.is_finite()
        && state.refractory_period.is_finite()
        && state.refractory_remaining.is_finite()
        && state.delta_t > 0.0
        && state.tau > 0.0
        && state.dt > 0.0
        && state.refractory_period >= 0.0
        && state.refractory_remaining >= 0.0
        && state.refractory_remaining <= state.refractory_period
        && state.v_threshold > state.v_rh
        && state.v < state.v_threshold
        && state.v_rest < state.v_threshold
        && state.v_reset < state.v_threshold
}

#[cfg(test)]
mod tests {
    use super::{validate_expif, ExpIFError, ExpIFNeuron};

    #[test]
    fn default_matches_the_deterministic_catalogue_contract() {
        let state = ExpIFNeuron::default();
        assert_eq!(state.v, -65.0);
        assert_eq!(state.v_threshold, 30.0);
        assert_eq!(state.v_rh, -59.9);
        assert_eq!(state.delta_t, 3.48);
        assert_eq!(state.tau, 10.0);
        assert_eq!(state.dt, 0.02);
        assert!(validate_expif(&state));
    }

    #[test]
    fn one_step_matches_an_independent_rk4_expansion() {
        let mut state = ExpIFNeuron::new();
        state.v = -62.0;
        state.dt = 0.05;
        let current = 5.0;
        let initial_v = state.v;
        let k1 = state.rhs(initial_v, current);
        let k2 = state.rhs(initial_v + 0.5 * state.dt * k1, current);
        let k3 = state.rhs(initial_v + 0.5 * state.dt * k2, current);
        let k4 = state.rhs(initial_v + state.dt * k3, current);
        let expected = initial_v + (state.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

        assert_eq!(state.step(current), Ok(0));
        assert!((state.v - expected).abs() < 1.0e-12);
    }

    #[test]
    fn enrolled_event_counts_are_exact() {
        for (current, expected) in [(0.0, 0), (5.0, 0), (20.0, 2)] {
            let mut state = ExpIFNeuron::new();
            let spikes: i32 = (0..1000).map(|_| state.step(current).unwrap()).sum();
            assert_eq!(spikes, expected, "current={current}");
        }
    }

    #[test]
    fn refractory_hold_uses_the_fitted_protocol_duration() {
        let mut state = ExpIFNeuron {
            refractory_period: 1.7,
            ..ExpIFNeuron::new()
        };
        while state.step(50.0).unwrap() == 0 {}
        assert_eq!(state.v, state.v_reset);
        assert_eq!(state.refractory_remaining, 1.7);
        for _ in 0..10 {
            assert_eq!(state.step(50.0), Ok(0));
            assert_eq!(state.v, state.v_reset);
        }
        assert!((state.refractory_remaining - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_input_and_state_are_mutation_free() {
        let mut state = ExpIFNeuron::new();
        let before = (state.v, state.refractory_remaining);
        assert_eq!(state.step(f64::INFINITY), Err(ExpIFError::InvalidInput));
        assert_eq!((state.v, state.refractory_remaining), before);

        state.refractory_remaining = 1.0;
        assert_eq!(state.step(0.0), Err(ExpIFError::InvalidState));
        assert_eq!(state.refractory_remaining, 1.0);
    }

    #[test]
    fn nonfinite_candidate_is_mutation_free() {
        let mut state = ExpIFNeuron::new();
        state.dt = 1.0e308;
        let before = (state.v, state.refractory_remaining);
        assert_eq!(state.step(1.0e308), Err(ExpIFError::NonFiniteUpdate));
        assert_eq!((state.v, state.refractory_remaining), before);
    }

    #[test]
    fn reset_restores_both_runtime_states() {
        let mut state = ExpIFNeuron {
            v: -60.0,
            refractory_period: 1.7,
            refractory_remaining: 0.8,
            ..ExpIFNeuron::new()
        };
        state.reset();
        assert_eq!(state.v, state.v_rest);
        assert_eq!(state.refractory_remaining, 0.0);
    }
}
