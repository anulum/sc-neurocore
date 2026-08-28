// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained upward-crossing Rulkov safety kernel

#[derive(Debug, Clone)]
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

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !validate(self) {
            return Err("invalid retained Rulkov runtime state");
        }
        if !current.is_finite() {
            return Err("invalid retained Rulkov current");
        }
        let previous_x = self.x;
        let boundary = self.alpha + self.y + current;
        if !boundary.is_finite() {
            return Err("invalid retained Rulkov branch boundary");
        }
        let x_next = if self.x <= 0.0 {
            let denominator = 1.0 - self.x;
            if denominator <= 0.0 || !denominator.is_finite() {
                return Err("invalid retained Rulkov denominator");
            }
            self.alpha / denominator + self.y + current
        } else if self.x < boundary {
            boundary
        } else {
            -1.0
        };
        let y_next = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        if !x_next.is_finite() || !y_next.is_finite() {
            return Err("invalid retained Rulkov candidate");
        }
        let event = i32::from(x_next >= self.x_threshold && previous_x < self.x_threshold);
        self.x = x_next;
        self.y = y_next;
        Ok(event)
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

pub fn validate(state: &SCUpwardCrossingRulkovMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.alpha.is_finite()
        && state.alpha > 0.0
        && state.sigma.is_finite()
        && state.mu.is_finite()
        && state.mu > 0.0
        && state.x_threshold.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_crossing_event_matches_project_contract() {
        let mut neuron = SCUpwardCrossingRulkovMapNeuron::new();
        assert_eq!(neuron.step(2.0).expect("finite candidate"), 1);
        assert_eq!(neuron.x, 1.0);
    }

    #[test]
    fn invalid_state_is_rejected_without_mutation() {
        let mut neuron = SCUpwardCrossingRulkovMapNeuron::new();
        neuron.y = f64::INFINITY;
        let before = (neuron.x, neuron.y);
        assert!(neuron.step(0.0).is_err());
        assert_eq!((neuron.x, neuron.y), before);
    }
}
