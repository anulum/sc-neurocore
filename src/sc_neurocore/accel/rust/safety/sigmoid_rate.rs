// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for sigmoid_rate

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct SigmoidRateNeuron {
    pub r: f64,
    pub tau: f64,
    pub beta: f64,
    pub theta: f64,
    pub dt: f64,
}

impl SigmoidRateNeuron {
    pub fn new() -> Self {
        Self {
            r: 0.0_f64,
            tau: 10.0_f64,
            beta: 1.0_f64,
            theta: 0.0_f64,
            dt: 0.1_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> Result<f64, &'static str> {
        if !i_ext.is_finite() || !validate_sigmoid_rate(self) {
            return Err("sigmoid-rate state/current must be finite and well-formed");
        }
        let sigma = sigmoid_rate_transfer(self.beta, i_ext, self.theta)?;
        let next_r = exact_relaxation(self.r, sigma, self.dt, self.tau);
        if !next_r.is_finite() || !(0.0..=1.0).contains(&next_r) {
            return Err("sigmoid-rate exact relaxation update became non-finite or left [0,1]");
        }
        self.r = next_r;
        Ok(next_r)
    }

    pub fn reset(&mut self) {
        // self.r = 0.0
        self.r = 0.0_f64;
        self.tau = 10.0_f64;
        self.beta = 1.0_f64;
        self.theta = 0.0_f64;
        self.dt = 0.1_f64;
    }
}

pub fn validate_sigmoid_rate(state: &SigmoidRateNeuron) -> bool {
    state.r.is_finite()
        && state.tau.is_finite()
        && state.beta.is_finite()
        && state.theta.is_finite()
        && state.dt.is_finite()
        && (0.0..=1.0).contains(&state.r)
        && state.tau > 0.0
        && state.dt > 0.0
}

fn exact_relaxation(r: f64, sigma: f64, dt: f64, tau: f64) -> f64 {
    let decay = (-dt / tau).exp();
    decay * r + (1.0 - decay) * sigma
}

fn sigmoid_rate_transfer(beta: f64, current: f64, theta: f64) -> Result<f64, &'static str> {
    let z = beta * (current - theta);
    if z.is_infinite() {
        return Ok(if z > 0.0 { 1.0 } else { 0.0 });
    }
    if !z.is_finite() {
        return Err("sigmoid-rate transfer argument must be finite or saturating");
    }
    if z >= 0.0 {
        Ok(1.0 / (1.0 + (-z).exp()))
    } else {
        let exp_z = z.exp();
        Ok(exp_z / (1.0 + exp_z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_reference(r: f64, sigma: f64, dt: f64, tau: f64) -> f64 {
        let decay = (-dt / tau).exp();
        decay * r + (1.0 - decay) * sigma
    }

    #[test]
    fn test_sigmoid_rate_new() {
        let state = SigmoidRateNeuron::new();
        assert!(validate_sigmoid_rate(&state));
    }

    #[test]
    fn test_sigmoid_rate_step() {
        let mut state = SigmoidRateNeuron::new();
        let rate = state.step(10.0).unwrap();
        assert!(rate > 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_sigmoid_rate_exact_relaxation_matches_reference() {
        let mut state = SigmoidRateNeuron {
            r: 0.25,
            tau: 10.0,
            beta: 2.0,
            theta: 1.0,
            dt: 0.5,
        };
        let sigma = sigmoid_rate_transfer(state.beta, 3.0, state.theta).unwrap();
        let expected = exact_reference(state.r, sigma, state.dt, state.tau);
        let rate = state.step(3.0).unwrap();
        assert!((rate - expected).abs() < 1.0e-12);
        assert!((state.r - expected).abs() < 1.0e-12);
    }

    #[test]
    fn test_sigmoid_rate_large_timestep_remains_bounded() {
        let mut state = SigmoidRateNeuron {
            r: 1.0,
            tau: 0.1,
            beta: 1.0,
            theta: 0.0,
            dt: 5.0,
        };
        let rate = state.step(-100.0).unwrap();
        assert!((0.0..=1.0).contains(&rate));
        assert!(rate < 1.0e-12);
    }

    #[test]
    fn test_sigmoid_rate_rejects_invalid_state_without_mutation() {
        let mut state = SigmoidRateNeuron::new();
        state.r = 0.5;
        state.r = 1.5;
        assert!(state.step(1.0).is_err());
        assert_eq!(state.r, 1.5);
    }

    #[test]
    fn test_sigmoid_rate_extreme_drive_saturates() {
        let mut state = SigmoidRateNeuron {
            beta: 1.0e308,
            ..SigmoidRateNeuron::new()
        };
        let high = state.step(1.0e308).unwrap();
        state.reset();
        let low = state.step(-1.0e308).unwrap();
        assert!(high > low);
        assert!((0.0..=1.0).contains(&high));
        assert!((0.0..=1.0).contains(&low));
    }
}
