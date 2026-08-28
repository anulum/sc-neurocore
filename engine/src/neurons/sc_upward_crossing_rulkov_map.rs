// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained upward-crossing Rulkov map

//! Count-neutral retained upward-crossing Rulkov-map identity.

/// Historical SC-NeuroCore Rulkov recurrence and event convention.
#[derive(Clone, Debug)]
pub struct SCUpwardCrossingRulkovMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub sigma: f64,
    pub mu: f64,
    pub x_threshold: f64,
}

impl SCUpwardCrossingRulkovMapNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.0,
            y: -3.0,
            alpha: 4.0,
            sigma: -1.6,
            mu: 0.001,
            x_threshold: 0.0,
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
            && self.x_threshold.is_finite()
    }

    fn candidate(&self, current: f64) -> Option<(f64, f64)> {
        let boundary = self.alpha + self.y + current;
        if !boundary.is_finite() {
            return None;
        }
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
            Some((x_new, y_new))
        } else {
            None
        }
    }

    /// Checked retained update; an invalid candidate leaves state unchanged.
    pub fn try_step(&mut self, current: f64) -> Option<i32> {
        if !self.valid_numeric_contract() || !current.is_finite() {
            return None;
        }
        let previous_x = self.x;
        let (x_new, y_new) = self.candidate(current)?;
        let event = i32::from(x_new >= self.x_threshold && previous_x < self.x_threshold);
        self.x = x_new;
        self.y = y_new;
        Some(event)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn simulate(&mut self, n_steps: usize, current: f64) -> (Vec<f64>, i64) {
        self.try_simulate(n_steps, current).unwrap_or_default()
    }

    /// Run a failure-atomic retained batch.
    pub fn try_simulate(&mut self, n_steps: usize, current: f64) -> Option<(Vec<f64>, i64)> {
        let mut candidate = self.clone();
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
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

impl Default for SCUpwardCrossingRulkovMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_event_precedes_source_reset_event() {
        let mut neuron = SCUpwardCrossingRulkovMapNeuron::new();
        assert_eq!(neuron.step(2.0), 1);
        assert_eq!(neuron.x, 1.0);
    }

    #[test]
    fn invalid_batch_preserves_state() {
        let mut neuron = SCUpwardCrossingRulkovMapNeuron::new();
        neuron.mu = 0.0;
        let before = (neuron.x, neuron.y);
        assert!(neuron.try_simulate(2, 0.0).is_none());
        assert_eq!((neuron.x, neuron.y), before);
    }
}
