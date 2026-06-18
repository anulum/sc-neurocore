// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for energy_lif

const ENERGY_LIF_V_MIN: f64 = -200.0_f64;
const ENERGY_LIF_V_MAX: f64 = 100.0_f64;
const ENERGY_LIF_GATE: f64 = 0.1_f64;

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
            return -1;
        }

        let (v_candidate, epsilon_candidate) = self.exact_candidate(i_ext);
        if !energy_lif_candidate_valid(v_candidate, epsilon_candidate, self.epsilon_0) {
            return -1;
        }
        if v_candidate >= self.v_threshold && epsilon_candidate > ENERGY_LIF_GATE {
            let epsilon_after_spike = (epsilon_candidate - self.alpha).max(0.0_f64);
            if !(epsilon_after_spike.is_finite() && epsilon_after_spike <= self.epsilon_0) {
                return -1;
            }
            self.v = self.v_reset;
            self.epsilon = epsilon_after_spike;
            return 1;
        }
        self.v = v_candidate;
        self.epsilon = epsilon_candidate;
        0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.epsilon = self.epsilon_0;
    }

    fn exact_candidate(&self, i_ext: f64) -> (f64, f64) {
        let membrane_decay = (-self.dt / self.tau_m).exp();
        let energy_decay = (-self.dt / self.tau_e).exp();
        let energy_delta = self.epsilon - self.epsilon_0;
        let epsilon_candidate = self.epsilon_0 + energy_delta * energy_decay;
        let steady_energy_integral = self.epsilon_0 * self.tau_m * (1.0_f64 - membrane_decay);
        let coupled_rate = (1.0_f64 / self.tau_m) - (1.0_f64 / self.tau_e);
        let transient_energy_integral = if coupled_rate.abs() < 1.0e-12_f64 {
            energy_delta * membrane_decay * self.dt
        } else {
            energy_delta * membrane_decay * (coupled_rate * self.dt).exp_m1() / coupled_rate
        };
        let v_candidate = self.v_rest
            + (self.v - self.v_rest) * membrane_decay
            + (self.resistance * i_ext / self.tau_m)
                * (steady_energy_integral + transient_energy_integral);
        (v_candidate, epsilon_candidate)
    }
}

pub fn validate_energy_lif(state: &EnergyLIFNeuron) -> bool {
    state.v.is_finite()
        && state.v >= ENERGY_LIF_V_MIN
        && state.v <= ENERGY_LIF_V_MAX
        && state.epsilon.is_finite()
        && state.epsilon >= 0.0
        && state.v_rest.is_finite()
        && state.v_reset.is_finite()
        && state.v_reset >= ENERGY_LIF_V_MIN
        && state.v_reset <= ENERGY_LIF_V_MAX
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

fn energy_lif_candidate_valid(v: f64, epsilon: f64, epsilon_0: f64) -> bool {
    v.is_finite()
        && v >= ENERGY_LIF_V_MIN
        && v <= ENERGY_LIF_V_MAX
        && epsilon.is_finite()
        && epsilon >= 0.0_f64
        && epsilon <= epsilon_0
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

    #[test]
    fn test_energy_lif_exact_candidate_commit() {
        let mut state = EnergyLIFNeuron::new();
        state.epsilon = 0.5_f64;
        let (v_candidate, epsilon_candidate) = state.exact_candidate(10.0_f64);
        assert_eq!(state.step(10.0_f64), 0);
        assert!((state.v - v_candidate).abs() < 1.0e-12_f64);
        assert!((state.epsilon - epsilon_candidate).abs() < 1.0e-12_f64);
    }

    #[test]
    fn test_energy_lif_invalid_state_does_not_mutate() {
        let mut state = EnergyLIFNeuron::new();
        state.epsilon = -1.0_f64;
        let before = state.clone();
        assert_eq!(state.step(10.0_f64), -1);
        assert_eq!(state.v, before.v);
        assert_eq!(state.epsilon, before.epsilon);
    }

    #[test]
    fn test_energy_lif_spike_uses_energy_candidate() {
        let mut state = EnergyLIFNeuron::new();
        let (_, epsilon_candidate) = state.exact_candidate(250.0_f64);
        assert_eq!(state.step(250.0_f64), 1);
        assert_eq!(state.v, state.v_reset);
        let expected = (epsilon_candidate - state.alpha).max(0.0_f64);
        assert!((state.epsilon - expected).abs() < 1.0e-12_f64);
    }
}
