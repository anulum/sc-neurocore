// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for the L9 holonomic memory adapter

#![allow(non_camel_case_types, non_snake_case)]

const DEFAULT_RNG_SEED: u64 = 0x4c39_4d45_4d5f_5453;

#[derive(Debug, Clone)]
pub struct L9_MemoryAdapter {
    pub n_memory_slots: usize,
    pub bitstream_length: usize,
    pub retrieval_gain: f64,
    pub weak_measurement_strength: f64,
    pub temporal_window: usize,
    pub rng_state: u64,
    pub imprints_psi: Vec<Vec<u8>>,
    pub retrieval_phi: Vec<Vec<u8>>,
    pub current_slot: usize,
}

impl L9_MemoryAdapter {
    pub fn new() -> Self {
        Self::try_new(64, 1024, 0.8, 0.1, 100)
            .expect("default L9 holonomic memory parameters are valid")
    }

    pub fn try_new(
        n_memory_slots: usize,
        bitstream_length: usize,
        retrieval_gain: f64,
        weak_measurement_strength: f64,
        temporal_window: usize,
    ) -> Result<Self, String> {
        validate_params(
            n_memory_slots,
            bitstream_length,
            retrieval_gain,
            weak_measurement_strength,
            temporal_window,
        )?;

        Ok(Self {
            n_memory_slots,
            bitstream_length,
            retrieval_gain,
            weak_measurement_strength,
            temporal_window,
            rng_state: DEFAULT_RNG_SEED,
            imprints_psi: vec![vec![0_u8; bitstream_length]; n_memory_slots],
            retrieval_phi: vec![vec![0_u8; bitstream_length]; n_memory_slots],
            current_slot: 0,
        })
    }

    pub fn encode(&mut self) -> Vec<u8> {
        let retrieval_prob = (self.total_overlap() * self.retrieval_gain).clamp(0.0, 1.0);
        let mut bitstream = vec![0_u8; self.bitstream_length];
        for bit in bitstream.iter_mut() {
            *bit = u8::from(next_unit_interval(&mut self.rng_state) < retrieval_prob);
        }
        bitstream
    }

    pub fn step_jax(&mut self, dt: f64, inputs: Option<&[Vec<f64>]>) -> Result<Vec<u8>, String> {
        validate_dt(dt)?;

        if let Some(rows) = inputs {
            let mapped = project_inputs(rows, self.n_memory_slots, self.bitstream_length)?;
            for slot in 0..self.n_memory_slots {
                for bit in 0..self.bitstream_length {
                    let psi_next = if mapped[slot][bit] > 0.5 {
                        1_u8
                    } else {
                        self.imprints_psi[slot][bit]
                    };
                    let weak_value_distance = (f64::from(psi_next) - 0.5).abs();
                    let phi_next = if weak_value_distance > self.weak_measurement_strength {
                        1_u8
                    } else {
                        self.retrieval_phi[slot][bit]
                    };
                    self.imprints_psi[slot][bit] = psi_next;
                    self.retrieval_phi[slot][bit] = phi_next;
                }
            }
        }

        Ok(self.encode())
    }

    pub fn decode(bitstreams: &[u8]) -> Result<f64, String> {
        if bitstreams.is_empty() {
            return Err("bitstreams must not be empty.".to_string());
        }
        let active = bitstreams.iter().filter(|bit| **bit > 0).count();
        Ok(active as f64 / bitstreams.len() as f64)
    }

    pub fn get_metrics(&self) -> (f64, f64) {
        let overlap = self.total_overlap() / self.n_memory_slots as f64;
        let imprint_density = self
            .imprints_psi
            .iter()
            .flatten()
            .map(|bit| f64::from(*bit))
            .sum::<f64>()
            / (self.n_memory_slots * self.bitstream_length) as f64;
        (overlap, imprint_density)
    }

    fn total_overlap(&self) -> f64 {
        self.imprints_psi
            .iter()
            .zip(self.retrieval_phi.iter())
            .map(|(psi_row, phi_row)| {
                psi_row
                    .iter()
                    .zip(phi_row.iter())
                    .map(|(psi, phi)| f64::from(*psi) * f64::from(*phi))
                    .sum::<f64>()
                    / self.bitstream_length as f64
            })
            .sum::<f64>()
    }
}

fn validate_params(
    n_memory_slots: usize,
    bitstream_length: usize,
    retrieval_gain: f64,
    weak_measurement_strength: f64,
    temporal_window: usize,
) -> Result<(), String> {
    if n_memory_slots == 0 {
        return Err("n_memory_slots must be positive.".to_string());
    }
    if bitstream_length == 0 {
        return Err("bitstream_length must be positive.".to_string());
    }
    if temporal_window == 0 {
        return Err("temporal_window must be positive.".to_string());
    }
    if !retrieval_gain.is_finite() || retrieval_gain < 0.0 {
        return Err("retrieval_gain must be finite and non-negative.".to_string());
    }
    if !weak_measurement_strength.is_finite() || !(0.0..=1.0).contains(&weak_measurement_strength) {
        return Err("weak_measurement_strength must be finite and in [0, 1].".to_string());
    }
    Ok(())
}

fn validate_dt(dt: f64) -> Result<(), String> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and positive.".to_string());
    }
    Ok(())
}

fn project_inputs(
    rows: &[Vec<f64>],
    n_memory_slots: usize,
    bitstream_length: usize,
) -> Result<Vec<Vec<f64>>, String> {
    if rows.is_empty() {
        return Err("inputs must contain at least one row.".to_string());
    }
    for row in rows {
        if row.len() != bitstream_length {
            return Err("inputs bitstream_length must match adapter parameters.".to_string());
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err("inputs must contain only finite values.".to_string());
        }
    }

    let mut mapped = Vec::with_capacity(n_memory_slots);
    for slot in 0..n_memory_slots {
        mapped.push(rows[slot % rows.len()].clone());
    }
    Ok(mapped)
}

fn next_unit_interval(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

pub fn validate_l9_mem(state: &L9_MemoryAdapter) -> bool {
    if validate_params(
        state.n_memory_slots,
        state.bitstream_length,
        state.retrieval_gain,
        state.weak_measurement_strength,
        state.temporal_window,
    )
    .is_err()
    {
        return false;
    }
    if state.current_slot >= state.n_memory_slots {
        return false;
    }
    if state.imprints_psi.len() != state.n_memory_slots
        || state.retrieval_phi.len() != state.n_memory_slots
    {
        return false;
    }
    state
        .imprints_psi
        .iter()
        .chain(state.retrieval_phi.iter())
        .all(|row| row.len() == state.bitstream_length && row.iter().all(|bit| *bit <= 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l9_mem_new() {
        let state = L9_MemoryAdapter::new();
        assert!(validate_l9_mem(&state));
    }

    #[test]
    fn test_l9_mem_rejects_invalid_parameters() {
        assert!(L9_MemoryAdapter::try_new(0, 1024, 0.8, 0.1, 100).is_err());
        assert!(L9_MemoryAdapter::try_new(64, 0, 0.8, 0.1, 100).is_err());
        assert!(L9_MemoryAdapter::try_new(64, 1024, -0.1, 0.1, 100).is_err());
        assert!(L9_MemoryAdapter::try_new(64, 1024, 0.8, 1.1, 100).is_err());
        assert!(L9_MemoryAdapter::try_new(64, 1024, 0.8, 0.1, 0).is_err());
    }

    #[test]
    fn test_l9_mem_tiles_mismatched_input_slots() {
        let mut state = L9_MemoryAdapter::try_new(4, 6, 0.8, 0.1, 100).unwrap();
        let inputs = vec![vec![0.0; 6], vec![1.0; 6]];

        let encoded = state.step_jax(0.05, Some(&inputs)).unwrap();

        assert_eq!(encoded.len(), 6);
        assert_eq!(state.imprints_psi[0], vec![0_u8; 6]);
        assert_eq!(state.imprints_psi[1], vec![1_u8; 6]);
        assert_eq!(state.imprints_psi[2], vec![0_u8; 6]);
        assert_eq!(state.imprints_psi[3], vec![1_u8; 6]);
    }

    #[test]
    fn test_l9_mem_rejects_malformed_inputs_without_mutation() {
        let mut state = L9_MemoryAdapter::try_new(2, 4, 0.8, 0.1, 100).unwrap();
        let before = state.imprints_psi.clone();

        assert!(state.step_jax(0.0, Some(&vec![vec![1.0; 4]])).is_err());
        assert!(state.step_jax(0.05, Some(&Vec::<Vec<f64>>::new())).is_err());
        assert!(state.step_jax(0.05, Some(&vec![vec![1.0; 3]])).is_err());
        assert!(state
            .step_jax(0.05, Some(&vec![vec![f64::NAN; 4]]))
            .is_err());
        assert_eq!(state.imprints_psi, before);
    }

    #[test]
    fn test_l9_mem_validator_rejects_corrupt_state() {
        let mut state = L9_MemoryAdapter::try_new(2, 4, 0.8, 0.1, 100).unwrap();
        state.current_slot = 2;
        assert!(!validate_l9_mem(&state));

        state.current_slot = 0;
        state.retrieval_phi[0].push(1);
        assert!(!validate_l9_mem(&state));
    }
}
