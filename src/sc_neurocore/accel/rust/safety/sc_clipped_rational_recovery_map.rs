// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Safety mirror for the retained rational-recovery map

#[derive(Debug, Clone)]
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

    /// Return -1 without mutation on invalid input or candidate.
    pub fn step(&mut self, current: f64) -> i32 {
        if !validate_sc_clipped_rational_recovery_map(self) || !current.is_finite() {
            return -1;
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
            return -1;
        }
        let x_new = x_candidate.clamp(-self.clip_bound, self.clip_bound);
        let y_new = y_candidate.clamp(-self.clip_bound, self.clip_bound);
        let event = i32::from(x_new >= self.x_threshold && x_previous < self.x_threshold);
        self.x = x_new;
        self.y = y_new;
        event
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

pub fn validate_sc_clipped_rational_recovery_map(
    state: &SCClippedRationalRecoveryMapNeuron,
) -> bool {
    [
        state.x,
        state.y,
        state.alpha,
        state.beta,
        state.j,
        state.x_threshold,
        state.clip_bound,
    ]
    .iter()
    .all(|value| value.is_finite())
        && state.alpha > 0.0
        && state.beta > 0.0
        && state.clip_bound > 0.0
        && state.x.abs() <= state.clip_bound
        && state.y.abs() <= state.clip_bound
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_first_step_is_retained() {
        let mut neuron = SCClippedRationalRecoveryMapNeuron::new();
        assert_eq!(neuron.step(0.0), 0);
        assert_eq!((neuron.x, neuron.y), (0.1, -0.001));
    }

    #[test]
    fn invalid_input_is_atomic() {
        let mut neuron = SCClippedRationalRecoveryMapNeuron::new();
        let before = (neuron.x, neuron.y);
        assert_eq!(neuron.step(f64::NAN), -1);
        assert_eq!((neuron.x, neuron.y), before);
    }
}
