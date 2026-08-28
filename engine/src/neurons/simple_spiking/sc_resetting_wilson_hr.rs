// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained resetting Wilson-HR project recurrence

//! Retained unit-capacitance Wilson-HR project recurrence with hard reset.

/// Historical SC-NeuroCore polynomial recurrence retained without paper attribution.
#[derive(Clone, Debug)]
pub struct SCResettingWilsonHRNeuron {
    /// Membrane state.
    pub v: f64,
    /// Recovery state.
    pub r: f64,
    /// Recovery time constant in milliseconds.
    pub tau_r: f64,
    /// Hard-reset event level.
    pub v_peak: f64,
    /// RK4 step in milliseconds.
    pub dt: f64,
}

impl SCResettingWilsonHRNeuron {
    /// Construct the historical project-default state and parameters.
    pub fn new() -> Self {
        Self {
            v: -0.7,
            r: 0.1,
            tau_r: 1.9,
            v_peak: 0.4,
            dt: 0.05,
        }
    }

    fn valid_numeric_contract(&self) -> bool {
        self.v.is_finite()
            && self.r.is_finite()
            && self.tau_r.is_finite()
            && self.tau_r > 0.0
            && self.v_peak.is_finite()
            && self.dt.is_finite()
            && self.dt > 0.0
    }

    fn poly(v: f64) -> f64 {
        -(17.81 + 47.71 * v + 32.63 * v * v) * (v - 0.55)
    }

    fn derivatives(&self, v: f64, r: f64, current: f64) -> Option<(f64, f64)> {
        if !(v.is_finite() && r.is_finite() && current.is_finite()) {
            return None;
        }
        let polynomial = Self::poly(v);
        let recovery_current = -26.0 * r * (v + 0.92);
        let dv = polynomial + recovery_current + current;
        let dr = (-r + 1.35 * v + 1.03) / self.tau_r;
        if polynomial.is_finite()
            && recovery_current.is_finite()
            && dv.is_finite()
            && dr.is_finite()
        {
            Some((dv, dr))
        } else {
            None
        }
    }

    fn rk4_candidate(&self, current: f64) -> Option<(f64, f64)> {
        let v0 = self.v;
        let r0 = self.r;
        let dt = self.dt;
        let k1 = self.derivatives(v0, r0, current)?;
        let k2 = self.derivatives(v0 + 0.5 * dt * k1.0, r0 + 0.5 * dt * k1.1, current)?;
        let k3 = self.derivatives(v0 + 0.5 * dt * k2.0, r0 + 0.5 * dt * k2.1, current)?;
        let k4 = self.derivatives(v0 + dt * k3.0, r0 + dt * k3.1, current)?;
        let next_v = v0 + dt * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0;
        let next_r = r0 + dt * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0;
        if next_v.is_finite() && next_r.is_finite() {
            Some((next_v, next_r))
        } else {
            None
        }
    }

    fn try_step(&mut self, current: f64) -> Option<i32> {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return None;
        }
        let (next_v, next_r) = self.rk4_candidate(current)?;
        let event = next_v >= self.v_peak;
        self.v = if event { -0.7 } else { next_v };
        self.r = next_r;
        Some(i32::from(event))
    }

    /// Advance one failure-atomic RK4 update and historical hard reset.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    /// Run one batch, returning an empty result without mutation on invalid arithmetic.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        self.try_simulate(n_steps, current).unwrap_or_default()
    }

    /// Run one failure-atomic batch, returning `None` on any invalid stage.
    pub fn try_simulate(&mut self, n_steps: usize, current: f64) -> Option<(Vec<f64>, i64)> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(candidate.try_step(current)?);
            trace.push(candidate.v);
        }
        *self = candidate;
        Some((trace, events))
    }

    /// Restore the historical dynamic state while retaining configuration.
    pub fn reset(&mut self) {
        self.v = -0.7;
        self.r = 0.1;
    }
}

impl Default for SCResettingWilsonHRNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_one_step_anchor_is_preserved() {
        let mut neuron = SCResettingWilsonHRNeuron::new();
        assert_eq!(neuron.step(2.0), 0);
        assert_eq!(neuron.v, -0.5988676025214146);
        assert_eq!(neuron.r, 0.10134793845659071);
    }

    #[test]
    fn batch_matches_repeated_step() {
        let mut batch = SCResettingWilsonHRNeuron::new();
        let mut repeated = SCResettingWilsonHRNeuron::new();
        let (trace, events) = batch.simulate(1_000, 2.0);
        let mut expected = Vec::with_capacity(1_000);
        let mut expected_events = 0_i64;
        for _ in 0..1_000 {
            expected_events += i64::from(repeated.step(2.0));
            expected.push(repeated.v);
        }
        assert_eq!(trace, expected);
        assert_eq!(events, expected_events);
        assert_eq!((batch.v, batch.r), (repeated.v, repeated.r));
    }

    #[test]
    fn invalid_batch_preserves_state() {
        let mut neuron = SCResettingWilsonHRNeuron {
            v: 1.0e103,
            ..Default::default()
        };
        let before = (neuron.v, neuron.r);
        assert!(neuron.try_simulate(2, 2.0).is_none());
        assert_eq!((neuron.v, neuron.r), before);
    }
}
