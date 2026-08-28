// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rulkov discrete map neuron

//! Rulkov discrete map neuron.

/// Rulkov (2002) piecewise nonlinear map for fast/slow bursting.
#[derive(Clone, Debug)]
pub struct RulkovMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub sigma: f64,
    pub mu: f64,
}

impl RulkovMapNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.0,
            y: -3.0,
            alpha: 4.0,
            sigma: -1.6,
            mu: 0.001,
        }
    }
    fn valid_numeric_contract(&self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.alpha.is_finite()
            && self.alpha > 0.0
            && self.sigma.is_finite()
            && self.mu.is_finite()
            && self.mu > 0.0
    }

    fn candidate(&self, current: f64) -> Option<(f64, f64, i32)> {
        let boundary = self.alpha + self.y + current;
        if !boundary.is_finite() {
            return None;
        }
        let event = i32::from(self.x > 0.0 && self.x >= boundary);
        let x_new = if self.x <= 0.0 {
            let denominator = 1.0 - self.x;
            if denominator <= 0.0 || !denominator.is_finite() {
                return None;
            }
            self.alpha / denominator + self.y + current
        } else if self.x < boundary {
            boundary
        } else {
            -1.0
        };
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        if x_new.is_finite() && y_new.is_finite() {
            Some((x_new, y_new, event))
        } else {
            None
        }
    }
    /// Checked source update; an invalid candidate leaves state unchanged.
    pub fn try_step(&mut self, current: f64) -> Option<i32> {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return None;
        }
        let (x_new, y_new, event) = self.candidate(current)?;
        self.x = x_new;
        self.y = y_new;
        Some(event)
    }
    /// Network-runner update; invalid input emits no event and preserves state.
    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }
    /// Run `n_steps` under a constant input, returning the `x` trace and the
    /// reset-branch event count. Reuses `step` so the trace is bit-identical to
    /// the per-step path and to the Python reference. The final state is left
    /// in `self.x` / `self.y`.
    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        self.try_simulate(n_steps, current).unwrap_or_default()
    }
    /// Run a failure-atomic source batch, returning `None` on any invalid stage.
    pub fn try_simulate(&mut self, n_steps: usize, current: f64) -> Option<(Vec<f64>, i64)> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut events: i64 = 0;
        for _ in 0..n_steps {
            events += i64::from(candidate.try_step(current)?);
            trace.push(candidate.x);
        }
        *self = candidate;
        Some((trace, events))
    }
    pub fn reset(&mut self) {
        self.x = -1.0;
        self.y = -3.0;
    }
}
impl Default for RulkovMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rulkov_fires() {
        let mut n = RulkovMapNeuron::new();
        let t: i32 = (0..2000).map(|_| n.step(0.5)).sum();
        assert!(t > 0);
    }

    #[test]
    fn source_event_is_rightmost_branch_execution() {
        let mut neuron = RulkovMapNeuron {
            x: 1.0,
            y: -3.0,
            ..Default::default()
        };
        assert_eq!(neuron.step(0.0), 1);
        assert_eq!(neuron.x, -1.0);
    }

    #[test]
    fn invalid_batch_is_failure_atomic() {
        let mut neuron = RulkovMapNeuron::new();
        neuron.alpha = f64::NAN;
        let before = (neuron.x, neuron.y);
        assert!(neuron.try_simulate(2, 0.0).is_none());
        assert_eq!((neuron.x, neuron.y), before);
    }
}
