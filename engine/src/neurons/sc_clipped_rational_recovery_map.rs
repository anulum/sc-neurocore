// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Retained clipped rational-recovery map

//! Count-neutral mirror of the retained project recurrence.

#[derive(Clone, Debug)]
pub struct SCClippedRationalRecoveryMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub beta: f64,
    pub j: f64,
    pub x_threshold: f64,
    pub clip_bound: f64,
}

impl SCClippedRationalRecoveryMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            alpha: 3.0,
            beta: 0.001,
            j: 0.1,
            x_threshold: 1.0,
            clip_bound: 1_000_000.0,
        }
    }

    fn parameters_are_valid(&self) -> bool {
        [
            self.x,
            self.y,
            self.alpha,
            self.beta,
            self.j,
            self.x_threshold,
            self.clip_bound,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.alpha > 0.0
            && self.beta > 0.0
            && self.clip_bound > 0.0
            && self.x.abs() <= self.clip_bound
            && self.y.abs() <= self.clip_bound
    }

    /// Checked simultaneous update. Rejected input leaves state unchanged.
    pub fn try_step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !self.parameters_are_valid() {
            return Err("invalid SC rational-recovery runtime state");
        }
        if !current.is_finite() {
            return Err("invalid SC rational-recovery current");
        }
        let x_previous = self.x;
        let field = if self.x < 0.0 {
            self.alpha * self.x
        } else {
            self.alpha * self.x / (1.0 + self.alpha * self.x)
        };
        let x_candidate = field + self.y + current + self.j;
        let y_candidate = self.y - self.beta * (self.x + 1.0);
        if !x_candidate.is_finite() || !y_candidate.is_finite() {
            return Err("SC rational-recovery candidate became non-finite");
        }
        let x_new = x_candidate.clamp(-self.clip_bound, self.clip_bound);
        let y_new = y_candidate.clamp(-self.clip_bound, self.clip_bound);
        let event = i32::from(x_new >= self.x_threshold && x_previous < self.x_threshold);
        self.x = x_new;
        self.y = y_new;
        Ok(event)
    }

    pub fn step(&mut self, current: f64) -> i32 {
        self.try_step(current).unwrap_or(0)
    }

    pub fn simulate(
        &mut self,
        n_steps: usize,
        current: f64,
    ) -> Result<(Vec<f64>, i64), &'static str> {
        let mut trace = Vec::with_capacity(n_steps);
        let mut events = 0_i64;
        for _ in 0..n_steps {
            events += i64::from(self.try_step(current)?);
            trace.push(self.x);
        }
        Ok((trace, events))
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}

impl Default for SCClippedRationalRecoveryMapNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_orbit_is_retained() {
        let mut neuron = SCClippedRationalRecoveryMapNeuron::new();
        assert_eq!(neuron.try_step(0.0).unwrap(), 0);
        assert_eq!((neuron.x, neuron.y), (0.1, -0.001));
    }

    #[test]
    fn invalid_input_is_atomic() {
        let mut neuron = SCClippedRationalRecoveryMapNeuron::new();
        let before = (neuron.x, neuron.y);
        assert!(neuron.try_step(f64::NAN).is_err());
        assert_eq!((neuron.x, neuron.y), before);
    }
}
