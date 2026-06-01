// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for leaky_compete_fire

#![allow(unused_variables, dead_code, non_snake_case)]

#[derive(Debug, Clone)]
pub struct LeakyCompeteFireNeuron {
    pub n_units: usize,
    pub v: Vec<f64>,
    pub tau: f64,
    pub v_threshold: f64,
    pub w_inh: f64,
    pub dt: f64,
}

impl LeakyCompeteFireNeuron {
    pub fn new() -> Self {
        Self {
            n_units: 4,
            v: vec![0.0_f64; 4],
            tau: 10.0_f64,
            v_threshold: 1.0_f64,
            w_inh: 0.5_f64,
            dt: 1.0_f64,
        }
    }

    pub fn step(&mut self, currents: &[f64]) -> Result<Vec<i32>, &'static str> {
        if currents.len() != self.n_units {
            return Err("LCF currents must match n_units");
        }
        if currents.iter().any(|current| !current.is_finite()) {
            return Err("LCF currents must contain only finite values");
        }
        validate_leaky_compete_fire(self)?;
        let decay = (-self.dt / self.tau).exp();
        let mut next_v = Vec::with_capacity(self.n_units);
        for (voltage, current) in self.v.iter().zip(currents.iter()) {
            let candidate = current + (voltage - current) * decay;
            if !candidate.is_finite() {
                return Err("LCF exact relaxation produced a non-finite candidate");
            }
            next_v.push(candidate);
        }
        let mut spikes = vec![0_i32; self.n_units];
        for i in 0..self.n_units {
            if next_v[i] >= self.v_threshold {
                spikes[i] = 1;
                next_v[i] = 0.0;
                for j in 0..self.n_units {
                    if j != i {
                        next_v[j] = (next_v[j] - self.w_inh).max(0.0);
                    }
                }
            }
        }
        self.v = next_v;
        Ok(spikes)
    }

    pub fn step_scalar(&mut self, current: f64) -> Result<Vec<i32>, &'static str> {
        let currents = vec![current; self.n_units];
        self.step(&currents)
    }

    pub fn reset(&mut self) {
        self.v = vec![0.0_f64; self.n_units];
    }
}

pub fn validate_leaky_compete_fire(state: &LeakyCompeteFireNeuron) -> Result<(), &'static str> {
    if state.n_units == 0 {
        return Err("LCF n_units must be positive");
    }
    if state.v.len() != state.n_units {
        return Err("LCF voltage vector length must match n_units");
    }
    if !(state.tau.is_finite() && state.tau > 0.0) {
        return Err("LCF tau must be finite and positive");
    }
    if !state.v_threshold.is_finite() {
        return Err("LCF threshold must be finite");
    }
    if !(state.w_inh.is_finite() && state.w_inh >= 0.0) {
        return Err("LCF inhibition weight must be finite and non-negative");
    }
    if !(state.dt.is_finite() && state.dt > 0.0) {
        return Err("LCF dt must be finite and positive");
    }
    if state.v.iter().any(|voltage| !voltage.is_finite()) {
        return Err("LCF voltage vector must contain only finite values");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_reference(voltage: f64, current: f64, tau: f64, dt: f64) -> f64 {
        current + (voltage - current) * (-dt / tau).exp()
    }

    #[test]
    fn test_leaky_compete_fire_new() {
        let state = LeakyCompeteFireNeuron::new();
        assert!(validate_leaky_compete_fire(&state).is_ok());
    }

    #[test]
    fn test_leaky_compete_fire_step() {
        let mut state = LeakyCompeteFireNeuron::new();
        let spikes = state.step(&[10.0, 10.0, 10.0, 10.0]).unwrap();
        assert_eq!(spikes.len(), 4);
        assert!(spikes.iter().all(|spike| *spike == 0 || *spike == 1));
    }

    #[test]
    fn exact_relaxation_matches_reference() {
        let mut state = LeakyCompeteFireNeuron {
            n_units: 3,
            v: vec![0.2, 0.4, 0.1],
            tau: 7.0,
            v_threshold: 100.0,
            w_inh: 0.5,
            dt: 2.5,
        };
        let currents = [1.0, 0.5, 0.0];
        let expected = [
            exact_reference(state.v[0], currents[0], state.tau, state.dt),
            exact_reference(state.v[1], currents[1], state.tau, state.dt),
            exact_reference(state.v[2], currents[2], state.tau, state.dt),
        ];

        let spikes = state.step(&currents).unwrap();

        assert_eq!(spikes, vec![0, 0, 0]);
        for (actual, expected_value) in state.v.iter().zip(expected.iter()) {
            assert!((*actual - *expected_value).abs() < 1.0e-12);
        }
    }

    #[test]
    fn invalid_state_does_not_mutate() {
        let mut state = LeakyCompeteFireNeuron {
            n_units: 2,
            v: vec![0.2, 0.4],
            tau: 0.0,
            v_threshold: 1.0,
            w_inh: 0.5,
            dt: 1.0,
        };
        let before = state.v.clone();

        assert!(state.step(&[1.0, 0.5]).is_err());
        assert_eq!(state.v, before);
    }
}
