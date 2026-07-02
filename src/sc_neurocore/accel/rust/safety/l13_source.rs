// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for L13 source-field dynamics

#![allow(non_camel_case_types, non_snake_case)]

#[derive(Debug, Clone)]
pub struct L13_SourceAdapter {
    pub n_vacuum_nodes: usize,
    pub bitstream_length: usize,
    pub j_primordial_coupling: f64,
    pub h_potential_bias: f64,
    pub lambda_scission: f64,
    pub rng_state: u64,
    pub vacuum_state: Vec<f64>,
    pub fim_density: Vec<f64>,
}

impl L13_SourceAdapter {
    pub fn new() -> Self {
        Self::try_new(256, 1024, 1.0, 0.01, 0.1)
            .expect("default L13 source-field parameters are valid")
    }

    pub fn try_new(
        n_vacuum_nodes: usize,
        bitstream_length: usize,
        j_primordial_coupling: f64,
        h_potential_bias: f64,
        lambda_scission: f64,
    ) -> Result<Self, String> {
        validate_params(
            n_vacuum_nodes,
            bitstream_length,
            j_primordial_coupling,
            h_potential_bias,
            lambda_scission,
        )?;

        let mut rng_state = 0x5343_504e_4c31_3353;
        let mut vacuum_state = vec![0.5; n_vacuum_nodes];
        if lambda_scission > 0.0 {
            let amplitude = lambda_scission.min(1.0) * 0.02;
            for value in vacuum_state.iter_mut() {
                let perturbation = (next_unit_interval(&mut rng_state) - 0.5) * amplitude;
                *value = (*value + perturbation).clamp(0.0, 1.0);
            }
        }

        Ok(Self {
            n_vacuum_nodes,
            bitstream_length,
            j_primordial_coupling,
            h_potential_bias,
            lambda_scission,
            rng_state,
            vacuum_state,
            fim_density: vec![0.0; n_vacuum_nodes],
        })
    }

    pub fn encode(&mut self) -> Vec<Vec<u8>> {
        let mut out = vec![vec![0_u8; self.bitstream_length]; self.n_vacuum_nodes];
        for (row, potential) in out.iter_mut().zip(self.vacuum_state.iter()) {
            let probability = potential.clamp(0.0, 1.0);
            for bit in row.iter_mut() {
                *bit = u8::from(next_unit_interval(&mut self.rng_state) < probability);
            }
        }
        out
    }

    pub fn vacuum_lattice_kernel(
        state: &[f64],
        coupling: f64,
        bias: f64,
        scission_rate: f64,
        feedback_drive: &[f64],
        dt: f64,
    ) -> Result<Vec<f64>, String> {
        validate_kernel_inputs(state, coupling, bias, scission_rate, feedback_drive, dt)?;

        let spin: Vec<f64> = state
            .iter()
            .map(|value| 2.0 * value.clamp(0.0, 1.0) - 1.0)
            .collect();
        let mut next_state = vec![0.0; state.len()];
        for index in 0..state.len() {
            let left = if index == 0 {
                spin.len() - 1
            } else {
                index - 1
            };
            let right = (index + 1) % spin.len();
            let neighbour_field = 0.5 * (spin[left] + spin[right]);
            let hamiltonian_drive =
                coupling * neighbour_field + bias + 0.25 * feedback_drive[index].clamp(-1.0, 1.0);
            let scission_drive =
                scission_rate * (spin[index] - spin[index] * spin[index] * spin[index]);
            let relaxation = -0.05 * spin[index];
            let spin_next = spin[index] + (hamiltonian_drive + scission_drive + relaxation) * dt;
            next_state[index] = (0.5 * (spin_next + 1.0)).clamp(0.0, 1.0);
        }
        Ok(next_state)
    }

    pub fn step_jax(
        &mut self,
        dt: f64,
        inputs: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<u8>>, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and positive.".to_string());
        }
        let feedback_drive = project_feedback(inputs, self.n_vacuum_nodes)?;
        let previous = self.vacuum_state.clone();
        self.vacuum_state = Self::vacuum_lattice_kernel(
            &self.vacuum_state,
            self.j_primordial_coupling,
            self.h_potential_bias,
            self.lambda_scission,
            &feedback_drive,
            dt,
        )?;

        let mut instant_fim = vec![0.0; self.n_vacuum_nodes];
        for index in 0..self.n_vacuum_nodes {
            let next = (index + 1) % self.n_vacuum_nodes;
            let variance =
                (self.vacuum_state[index] * (1.0 - self.vacuum_state[index])).max(1.0e-6);
            let temporal_delta = self.vacuum_state[index] - previous[index];
            let lattice_delta = self.vacuum_state[next] - self.vacuum_state[index];
            instant_fim[index] =
                (temporal_delta * temporal_delta + lattice_delta * lattice_delta) / variance;
        }
        for (density, instant) in self.fim_density.iter_mut().zip(instant_fim.iter()) {
            *density = 0.9 * *density + 0.1 * *instant;
        }
        Ok(self.encode())
    }

    pub fn decode(bitstreams: &[Vec<f64>]) -> Result<f64, String> {
        if bitstreams.is_empty() || bitstreams.iter().any(Vec::is_empty) {
            return Err("bitstreams must be a non-empty matrix.".to_string());
        }
        if bitstreams
            .iter()
            .flat_map(|row| row.iter())
            .any(|value| !value.is_finite())
        {
            return Err("bitstreams must contain only finite values.".to_string());
        }
        let total_bits = bitstreams.iter().map(Vec::len).sum::<usize>();
        let active_bits = bitstreams
            .iter()
            .flat_map(|row| row.iter())
            .filter(|value| **value != 0.0)
            .count();
        Ok(active_bits as f64 / total_bits as f64)
    }

    pub fn get_metrics(&self) -> (f64, f64) {
        (mean(&self.vacuum_state), mean(&self.fim_density))
    }
}

fn validate_params(
    n_vacuum_nodes: usize,
    bitstream_length: usize,
    j_primordial_coupling: f64,
    h_potential_bias: f64,
    lambda_scission: f64,
) -> Result<(), String> {
    if n_vacuum_nodes == 0 {
        return Err("n_vacuum_nodes must be positive.".to_string());
    }
    if bitstream_length == 0 {
        return Err("bitstream_length must be positive.".to_string());
    }
    if !j_primordial_coupling.is_finite() {
        return Err("j_primordial_coupling must be finite.".to_string());
    }
    if !h_potential_bias.is_finite() {
        return Err("h_potential_bias must be finite.".to_string());
    }
    if !lambda_scission.is_finite() || lambda_scission < 0.0 {
        return Err("lambda_scission must be finite and non-negative.".to_string());
    }
    Ok(())
}

fn validate_kernel_inputs(
    state: &[f64],
    coupling: f64,
    bias: f64,
    scission_rate: f64,
    feedback_drive: &[f64],
    dt: f64,
) -> Result<(), String> {
    if state.is_empty() || state.len() != feedback_drive.len() {
        return Err("state and feedback dimensions must be equal and non-empty.".to_string());
    }
    if state.iter().any(|value| !value.is_finite())
        || feedback_drive.iter().any(|value| !value.is_finite())
    {
        return Err("state and feedback must be finite.".to_string());
    }
    validate_params(state.len(), 1, coupling, bias, scission_rate)?;
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and positive.".to_string());
    }
    Ok(())
}

fn project_feedback(
    inputs: Option<&[Vec<f64>]>,
    n_vacuum_nodes: usize,
) -> Result<Vec<f64>, String> {
    let Some(rows) = inputs else {
        return Ok(vec![0.0; n_vacuum_nodes]);
    };
    if rows.is_empty() {
        return Err("inputs must contain at least one row.".to_string());
    }

    let mut raw = Vec::with_capacity(rows.len());
    for row in rows {
        if row.is_empty() {
            return Err("inputs must contain at least one column.".to_string());
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err("inputs must contain only finite values.".to_string());
        }
        raw.push(row.iter().sum::<f64>() / row.len() as f64);
    }

    let projected = if raw.len() == n_vacuum_nodes {
        raw
    } else {
        vec![mean(&raw); n_vacuum_nodes]
    };
    Ok(projected
        .iter()
        .map(|value| (2.0 * value - 1.0).clamp(-1.0, 1.0))
        .collect())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn next_unit_interval(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

pub fn validate_l13_source(state: &L13_SourceAdapter) -> bool {
    if validate_params(
        state.n_vacuum_nodes,
        state.bitstream_length,
        state.j_primordial_coupling,
        state.h_potential_bias,
        state.lambda_scission,
    )
    .is_err()
    {
        return false;
    }
    state.vacuum_state.len() == state.n_vacuum_nodes
        && state.fim_density.len() == state.n_vacuum_nodes
        && state
            .vacuum_state
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        && state.fim_density.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l13_source_new() {
        let state = L13_SourceAdapter::new();
        assert!(validate_l13_source(&state));
    }

    #[test]
    fn test_l13_source_rejects_invalid_parameters() {
        assert!(L13_SourceAdapter::try_new(0, 1024, 1.0, 0.01, 0.1).is_err());
        assert!(L13_SourceAdapter::try_new(256, 0, 1.0, 0.01, 0.1).is_err());
        assert!(L13_SourceAdapter::try_new(256, 1024, f64::NAN, 0.01, 0.1).is_err());
        assert!(L13_SourceAdapter::try_new(256, 1024, 1.0, f64::INFINITY, 0.1).is_err());
        assert!(L13_SourceAdapter::try_new(256, 1024, 1.0, 0.01, -0.1).is_err());
    }

    #[test]
    fn test_l13_source_broadcasts_mismatched_feedback_rows() {
        let mut state = L13_SourceAdapter::try_new(4, 6, 0.0, 0.0, 0.0).unwrap();
        let inputs = vec![vec![0.0; 6], vec![1.0; 6]];

        let encoded = state.step_jax(0.05, Some(&inputs)).unwrap();

        assert_eq!(encoded.len(), 4);
        assert!(state
            .vacuum_state
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() < 1.0e-12));
    }

    #[test]
    fn test_l13_source_local_lattice_coupling_lifts_neighbours() {
        let feedback = vec![0.0; 7];
        let next = L13_SourceAdapter::vacuum_lattice_kernel(
            &[0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5],
            1.0,
            0.0,
            0.0,
            &feedback,
            0.05,
        )
        .unwrap();

        let neighbour_lift = (next[2] - 0.5) + (next[4] - 0.5);
        let far_lift = (next[0] - 0.5) + (next[6] - 0.5);
        assert!(neighbour_lift > far_lift);
    }

    #[test]
    fn test_l13_source_rejects_malformed_inputs_without_mutation() {
        let mut state = L13_SourceAdapter::try_new(3, 4, 1.0, 0.01, 0.1).unwrap();
        let before_vacuum = state.vacuum_state.clone();
        let before_fim = state.fim_density.clone();

        assert!(state.step_jax(0.0, Some(&vec![vec![1.0; 4]])).is_err());
        assert!(state.step_jax(0.05, Some(&Vec::<Vec<f64>>::new())).is_err());
        assert!(state
            .step_jax(0.05, Some(&vec![Vec::<f64>::new()]))
            .is_err());
        assert!(state
            .step_jax(0.05, Some(&vec![vec![f64::NAN; 4]]))
            .is_err());

        assert_eq!(state.vacuum_state, before_vacuum);
        assert_eq!(state.fim_density, before_fim);
    }

    #[test]
    fn test_l13_source_decode_rejects_malformed_bitstreams() {
        assert!(L13_SourceAdapter::decode(&Vec::<Vec<f64>>::new()).is_err());
        assert!(L13_SourceAdapter::decode(&vec![Vec::<f64>::new()]).is_err());
        assert!(L13_SourceAdapter::decode(&vec![vec![f64::INFINITY]]).is_err());
        assert_eq!(
            L13_SourceAdapter::decode(&vec![vec![1.0, 0.0]]).unwrap(),
            0.5
        );
    }

    #[test]
    fn test_l13_source_validator_rejects_corrupt_state() {
        let mut state = L13_SourceAdapter::try_new(3, 4, 1.0, 0.01, 0.1).unwrap();
        state.vacuum_state[0] = f64::NAN;
        assert!(!validate_l13_source(&state));

        let mut wrong_shape = L13_SourceAdapter::try_new(3, 4, 1.0, 0.01, 0.1).unwrap();
        wrong_shape.fim_density.pop();
        assert!(!validate_l13_source(&wrong_shape));
    }
}
