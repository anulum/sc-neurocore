// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for gamma_motor_neuron

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct GammaMotorNeuron {
    pub v: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau: f64,
    pub adapt: f64,
    pub tau_adapt: f64,
    pub a_adapt: f64,
    pub gain: f64,
    pub dynamic: f64,
    pub dt: f64,
}

impl GammaMotorNeuron {
    pub fn new() -> Self {
        Self {
            v: -65.0_f64,
            v_rest: -65.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau: 8.0_f64,
            adapt: 0.0_f64,
            tau_adapt: 100.0_f64,
            a_adapt: 0.3_f64,
            gain: 1.0_f64,
            dynamic: 1.0_f64,
            dt: 0.5_f64,
        }
    }

    pub fn static_type(&self) -> f64 {
        0.0
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_gamma_motor_neuron(self) || !i_ext.is_finite() {
            return 0;
        }
        let v_old = self.v;
        let adapt_old = self.adapt;
        let input = self.gain * i_ext.max(0.0) - adapt_old;
        let v_target = self.v_rest + input;
        let v_candidate = v_target + (v_old - v_target) * (-self.dt / self.tau).exp();
        let adapt_target = self.a_adapt * (v_candidate - self.v_rest);
        let adapt_candidate =
            adapt_target + (adapt_old - adapt_target) * (-self.dt / self.tau_adapt).exp();
        if !v_candidate.is_finite() || !adapt_candidate.is_finite() {
            return 0;
        }
        self.v = v_candidate;
        self.adapt = adapt_candidate;
        if self.v >= self.v_threshold {
            self.v = self.v_reset;
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.adapt = 0.0_f64;
    }
}

pub fn validate_gamma_motor_neuron(state: &GammaMotorNeuron) -> bool {
    [
        state.v,
        state.v_rest,
        state.v_reset,
        state.v_threshold,
        state.tau,
        state.adapt,
        state.tau_adapt,
        state.a_adapt,
        state.gain,
        state.dynamic,
        state.dt,
    ]
    .iter()
    .all(|value| value.is_finite())
        && state.tau > 0.0
        && state.tau_adapt > 0.0
        && state.dt > 0.0
        && state.gain >= 0.0
        && state.v_reset < state.v_threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_motor_neuron_new() {
        let state = GammaMotorNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_gamma_motor_neuron(&state));
    }

    #[test]
    fn test_gamma_motor_neuron_step() {
        let mut state = GammaMotorNeuron::new();
        let spike = state.step(20.0);
        assert!(spike == 0 || spike == 1);
        assert!(state.v.is_finite());
        assert!(state.adapt.is_finite());
    }

    #[test]
    fn test_gamma_motor_neuron_invalid_drive_preserves_state() {
        let mut state = GammaMotorNeuron::new();
        let before = state.clone();
        assert_eq!(state.step(f64::NAN), 0);
        assert_eq!(state.v, before.v);
        assert_eq!(state.adapt, before.adapt);
    }
}
