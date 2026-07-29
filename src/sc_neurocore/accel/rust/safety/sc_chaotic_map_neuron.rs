// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for the SC two-state chaotic map

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct SCChaoticMapNeuron {
    pub x: f64,
    pub y: f64,
    pub k_f: f64,
    pub k_s: f64,
    pub alpha: f64,
    pub delta: f64,
    pub x_threshold: f64,
}

impl SCChaoticMapNeuron {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            k_f: 0.7,
            k_s: 0.95,
            alpha: 2.0,
            delta: 0.05,
            x_threshold: 0.5,
        }
    }

    pub fn step(&mut self, current: f64) -> Result<i32, &'static str> {
        if !validate_sc_chaotic_map_neuron(self) || !current.is_finite() {
            return Err("invalid SC chaotic map state, parameters, or current");
        }
        let previous = self.x;
        let x_next = self.k_f * self.x * logistic(self.x + self.alpha) - self.y + current;
        let y_next = self.k_s * self.y + self.delta * self.x;
        if !x_next.is_finite() || !y_next.is_finite() {
            return Err("non-finite SC chaotic map candidate");
        }
        self.x = x_next.clamp(-10.0, 10.0);
        self.y = y_next.clamp(-10.0, 10.0);
        Ok(i32::from(
            previous < self.x_threshold && self.x >= self.x_threshold,
        ))
    }

    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
    }
}

pub fn validate_sc_chaotic_map_neuron(state: &SCChaoticMapNeuron) -> bool {
    state.x.is_finite()
        && state.y.is_finite()
        && state.k_f.is_finite()
        && state.k_f >= 0.0
        && state.k_s.is_finite()
        && state.alpha.is_finite()
        && state.delta.is_finite()
        && state.delta >= 0.0
        && state.x_threshold.is_finite()
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_recurrence_is_checked() {
        let mut neuron = SCChaoticMapNeuron::new();
        assert_eq!(neuron.step(10.0).unwrap(), 1);
        assert!(neuron.x.is_finite());
        assert!(neuron.y.is_finite());
    }

    #[test]
    fn rejected_input_is_atomic() {
        let mut neuron = SCChaoticMapNeuron::new();
        assert!(neuron.step(f64::NAN).is_err());
        assert_eq!((neuron.x, neuron.y), (0.0, 0.0));
    }
}
