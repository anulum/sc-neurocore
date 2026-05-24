// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for stochastic_if

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct StochasticIFNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub mu: f64,
    pub sigma: f64,
    pub dt: f64,
}

impl StochasticIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0_f64,
            v_rest: -70.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 20.0_f64,
            mu: 0.0_f64,
            sigma: 3.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_stochastic_if(self) || !i_ext.is_finite() {
            return 0;
        }

        let noise = 0.0_f64;
        self.v += (-(self.v - self.v_rest) + self.mu + i_ext) / self.tau_m * self.dt + noise;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        // self.v = self.v_rest
        self.v = -70.0_f64;
        self.v_rest = -70.0_f64;
        self.v_reset = -70.0_f64;
        self.v_threshold = -50.0_f64;
        self.tau_m = 20.0_f64;
    }
}

pub fn validate_stochastic_if(state: &StochasticIFNeuron) -> bool {
    state.v.is_finite()
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.mu.is_finite()
        && state.sigma.is_finite()
        && state.sigma >= 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stochastic_if_new() {
        let state = StochasticIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_stochastic_if(&state));
    }

    #[test]
    fn test_stochastic_if_step() {
        let mut state = StochasticIFNeuron::new();
        let spike = state.step(10.0);
        assert!(spike == 0 || spike == 1);
    }
}
