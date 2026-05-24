// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for fractional_lif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct FractionalLIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub alpha: f64,
    pub resistance: f64,
    pub dt: f64,
    pub max_history: usize,
    pub history: Vec<f64>,
    pub gl_coeffs: Vec<f64>,
}

impl FractionalLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: 0.0_f64,
            v_rest: 0.0_f64,
            v_reset: 0.0_f64,
            v_threshold: 1.0_f64,
            alpha: 0.8_f64,
            resistance: 1.0_f64,
            dt: 1.0_f64,
            max_history: 100_usize,
            history: vec![0.0_f64; 100],
            gl_coeffs: compute_gl_coefficients(0.8_f64, 100_usize),
        }
    }

    pub fn recompute_gl_coefficients(&mut self) {
        self.gl_coeffs = compute_gl_coefficients(self.alpha, self.max_history);
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_fractional_lif(self) || !i_ext.is_finite() {
            return 0;
        }

        let rhs = -(self.v - self.v_rest) + self.resistance * i_ext;
        let terms = self
            .history
            .len()
            .min(self.max_history)
            .min(self.gl_coeffs.len());
        let mut gl_sum = 0.0_f64;
        for k in 1..terms {
            gl_sum += self.gl_coeffs[k] * self.history[self.history.len() - k];
        }
        self.v = rhs * self.dt.powf(self.alpha) - gl_sum;
        self.history.push(self.v);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            if let Some(last) = self.history.last_mut() {
                *last = self.v_reset;
            }
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.history = vec![self.v_rest; self.max_history];
    }
}

pub fn compute_gl_coefficients(alpha: f64, max_history: usize) -> Vec<f64> {
    let mut coeffs = vec![1.0_f64];
    for k in 1..max_history {
        let previous = coeffs[k - 1];
        coeffs.push(previous * (k as f64 - 1.0 - alpha) / k as f64);
    }
    coeffs
}

pub fn validate_fractional_lif(state: &FractionalLIFNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.alpha.is_finite()
        && state.alpha > 0.0
        && state.alpha <= 1.0
        && state.resistance.is_finite()
        && state.resistance > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.max_history > 1
        && state.history.len() == state.max_history
        && state.gl_coeffs.len() == state.max_history
        && state.history.iter().all(|value| value.is_finite())
        && state.gl_coeffs.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractional_lif_new() {
        let state = FractionalLIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_fractional_lif(&state));
    }

    #[test]
    fn test_fractional_lif_step() {
        let mut state = FractionalLIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_fractional_lif_alpha_one_matches_euler_limit() {
        let mut state = FractionalLIFNeuron::new();
        state.v = 0.25;
        state.alpha = 1.0;
        state.dt = 0.1;
        state.history = vec![0.0; 99];
        state.history.push(0.25);
        state.recompute_gl_coefficients();

        assert_eq!(state.step(0.5), 0);
        assert!((state.v - 0.275).abs() < 1.0e-12);
    }
}
