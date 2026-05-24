// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for energy_lif

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct EnergyLIFNeuron {
    pub v: f64,
    pub epsilon: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub v_threshold: f64,
    pub tau_m: f64,
    pub tau_e: f64,
    pub alpha: f64,
    pub epsilon_0: f64,
    pub resistance: f64,
    pub dt: f64,
}

impl EnergyLIFNeuron {
    pub fn new() -> Self {
        Self {
            v: -70.0_f64,
            epsilon: 1.0_f64,
            v_rest: -70.0_f64,
            v_reset: -70.0_f64,
            v_threshold: -50.0_f64,
            tau_m: 10.0_f64,
            tau_e: 500.0_f64,
            alpha: 0.1_f64,
            epsilon_0: 1.0_f64,
            resistance: 1.0_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, i_ext: f64) -> i32 {
        if !validate_energy_lif(self) || !i_ext.is_finite() {
            return 0;
        }

        let effective_r = self.resistance * self.epsilon;
        self.v += (-(self.v - self.v_rest) + effective_r * i_ext) / self.tau_m * self.dt;
        self.epsilon += (self.epsilon_0 - self.epsilon) / self.tau_e * self.dt;
        if self.v >= self.v_threshold && self.epsilon > 0.1 {
            self.v = self.v_reset;
            self.epsilon = (self.epsilon - self.alpha).max(0.0);
            return 1;
        }
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.epsilon = self.epsilon_0;
    }
}

pub fn validate_energy_lif(state: &EnergyLIFNeuron) -> bool {
    state.v.is_finite()
        && state.epsilon.is_finite()
        && state.epsilon >= 0.0
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_threshold.is_finite()
        && state.tau_m.is_finite()
        && state.tau_m > 0.0
        && state.tau_e.is_finite()
        && state.tau_e > 0.0
        && state.alpha.is_finite()
        && state.alpha >= 0.0
        && state.epsilon_0.is_finite()
        && state.epsilon_0 >= 0.0
        && state.resistance.is_finite()
        && state.resistance > 0.0
        && state.dt.is_finite()
        && state.dt > 0.0
        && state.epsilon <= state.epsilon_0
        && state.dt <= state.tau_m
        && state.dt <= state.tau_e
        && state.v_threshold > state.v_rest
        && state.v_threshold > state.v_reset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_lif_new() {
        let state = EnergyLIFNeuron::new();
        assert!(state.v.is_finite());
        assert!(validate_energy_lif(&state));
    }

    #[test]
    fn test_energy_lif_step() {
        let mut state = EnergyLIFNeuron::new();
        let spike = state.step(30.0);
        assert!(spike == 0 || spike == 1);
    }

    #[test]
    fn test_energy_lif_rejects_overfilled_reserve() {
        let mut state = EnergyLIFNeuron::new();
        state.epsilon = 1.1;
        assert!(!validate_energy_lif(&state));
    }
}
