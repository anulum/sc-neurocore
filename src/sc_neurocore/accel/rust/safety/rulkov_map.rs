// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for rulkov_map

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct RulkovMapNeuron {
    pub x: f64,
    pub y: f64,
    pub alpha: f64,
    pub sigma: f64,
    pub mu: f64,
    pub x_threshold: f64,
}

impl RulkovMapNeuron {
    pub fn new() -> Self {
        Self {
            x: -1.0_f64,
            y: -3.0_f64,
            alpha: 4.0_f64,
            sigma: -1.6_f64,
            mu: 0.001_f64,
            x_threshold: 0.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<i32, &'static str> {
        if !validate_rulkov_map(self) {
            return Err("invalid Rulkov map runtime state");
        }
        if !i_ext.is_finite() {
            return Err("invalid Rulkov map current");
        }

        let x_prev = self.x;
        let branch_boundary = self.alpha + self.y + i_ext;
        if !branch_boundary.is_finite() {
            return Err("invalid Rulkov map branch boundary");
        }
        let x_new = if self.x <= 0.0 {
            let denominator = 1.0 - self.x;
            if denominator <= 0.0 || !denominator.is_finite() {
                return Err("invalid Rulkov map branch denominator");
            }
            self.alpha / denominator + self.y + i_ext
        } else if self.x < branch_boundary {
            branch_boundary
        } else {
            -1.0
        };
        let y_new = self.y - self.mu * (self.x + 1.0) + self.mu * self.sigma;
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("invalid Rulkov map candidate state");
        }

        self.x = x_new;
        self.y = y_new;
        Ok(if self.x >= self.x_threshold && x_prev < self.x_threshold {
            1
        } else {
            0
        })
    }

    pub fn reset(&mut self) {
        // self.x, self.y = -1.0, -3.0
        self.x = -1.0_f64;
        self.y = -3.0_f64;
        self.alpha = 4.0_f64;
        self.sigma = -1.6_f64;
        self.mu = 0.001_f64;
    }
}

pub fn validate_rulkov_map(state: &RulkovMapNeuron) -> bool {
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
    fn test_rulkov_map_new() {
        let state = RulkovMapNeuron::new();
        assert!(validate_rulkov_map(&state));
    }

    #[test]
    fn test_rulkov_map_step() {
        let mut state = RulkovMapNeuron::new();
        let spike = state.step(10.0).unwrap();
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_rulkov_map_rejects_invalid_runtime_state() {
        let mut state = RulkovMapNeuron::new();
        state.y = f64::INFINITY;
        assert!(state.step(1.0).is_err());
    }
}
