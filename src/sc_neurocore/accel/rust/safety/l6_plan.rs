// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for L6 Gaia-field dynamics

#![allow(non_camel_case_types, non_snake_case)]

#[derive(Debug, Clone)]
pub struct L6_PlanetaryAdapter {
    pub n_regions: usize,
    pub bitstream_length: usize,
    pub f_schumann: f64,
    pub q_factor: f64,
    pub alpha_gaia: f64,
    pub p_percolation: f64,
    pub rng_state: u64,
    pub phi_planetary: Vec<f64>,
    pub regional_coherence: Vec<f64>,
    pub t: f64,
}

impl L6_PlanetaryAdapter {
    pub fn new() -> Self {
        Self::try_new(100, 1024, 7.83, 4.0, 0.05, 0.592)
            .expect("default L6 planetary parameters are valid")
    }

    pub fn try_new(
        n_regions: usize,
        bitstream_length: usize,
        f_schumann: f64,
        q_factor: f64,
        alpha_gaia: f64,
        p_percolation: f64,
    ) -> Result<Self, String> {
        validate_params(
            n_regions,
            bitstream_length,
            f_schumann,
            q_factor,
            alpha_gaia,
            p_percolation,
        )?;

        Ok(Self {
            n_regions,
            bitstream_length,
            f_schumann,
            q_factor,
            alpha_gaia,
            p_percolation,
            rng_state: 0x5343_504e_4c36_4741,
            phi_planetary: vec![0.0; n_regions],
            regional_coherence: vec![0.1; n_regions],
            t: 0.0,
        })
    }

    pub fn encode(&mut self) -> Vec<Vec<u8>> {
        let mut out = vec![vec![0_u8; self.bitstream_length]; self.n_regions];
        for (row, coherence) in out.iter_mut().zip(self.regional_coherence.iter()) {
            let probability = coherence.clamp(0.0, 1.0);
            for bit in row.iter_mut() {
                *bit = u8::from(next_unit_interval(&mut self.rng_state) < probability);
            }
        }
        out
    }

    pub fn gaia_kernel(
        phi: &[f64],
        sync_inputs: &[f64],
        alpha: f64,
        freq: f64,
        q_factor: f64,
        p_percolation: f64,
        t: f64,
        dt: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        validate_kernel_inputs(
            phi,
            sync_inputs,
            alpha,
            freq,
            q_factor,
            p_percolation,
            t,
            dt,
        )?;

        let bounded_sync: Vec<f64> = sync_inputs
            .iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect();
        let order_parameter = mean(&bounded_sync).clamp(0.0, 1.0);
        let driver = (2.0 * std::f64::consts::PI * freq * t).cos();
        let superradiant_gain = 1.0 + q_factor * order_parameter * order_parameter;

        let mut phi_next = vec![0.0; phi.len()];
        let mut coherence_next = vec![0.0; phi.len()];
        let percolation_gate = logistic(q_factor * (order_parameter - p_percolation));

        for index in 0..phi.len() {
            let d_phi =
                alpha * bounded_sync[index] * superradiant_gain * driver - 0.05 * phi[index];
            let next_phi = phi[index] + d_phi * dt;
            let local_field_activation = 1.0 - (-q_factor * next_phi.abs()).exp();
            phi_next[index] = next_phi;
            coherence_next[index] = (percolation_gate * local_field_activation).clamp(0.0, 1.0);
        }

        Ok((phi_next, coherence_next))
    }

    pub fn step_jax(
        &mut self,
        dt: f64,
        inputs: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<u8>>, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and positive.".to_string());
        }
        let sync_drive = project_inputs(inputs, self.n_regions, self.bitstream_length)?;
        self.t += dt;
        let (phi, coherence) = Self::gaia_kernel(
            &self.phi_planetary,
            &sync_drive,
            self.alpha_gaia,
            self.f_schumann,
            self.q_factor,
            self.p_percolation,
            self.t,
            dt,
        )?;
        self.phi_planetary = phi;
        self.regional_coherence = coherence;
        Ok(self.encode())
    }

    pub fn decode(bitstreams: &[Vec<u8>]) -> Result<f64, String> {
        if bitstreams.is_empty() || bitstreams.iter().any(|row| row.is_empty()) {
            return Err("bitstreams must be a non-empty matrix.".to_string());
        }
        let total_bits = bitstreams.iter().map(Vec::len).sum::<usize>();
        let active_bits = bitstreams
            .iter()
            .flat_map(|row| row.iter())
            .filter(|bit| **bit != 0)
            .count();
        Ok(active_bits as f64 / total_bits as f64)
    }

    pub fn get_metrics(&self) -> (f64, f64, f64) {
        (
            mean(&self.phi_planetary),
            mean(&self.regional_coherence),
            (self.t * self.f_schumann).rem_euclid(1.0),
        )
    }
}

fn validate_params(
    n_regions: usize,
    bitstream_length: usize,
    f_schumann: f64,
    q_factor: f64,
    alpha_gaia: f64,
    p_percolation: f64,
) -> Result<(), String> {
    if n_regions == 0 {
        return Err("n_regions must be positive.".to_string());
    }
    if bitstream_length == 0 {
        return Err("bitstream_length must be positive.".to_string());
    }
    for (name, value) in [
        ("f_schumann", f_schumann),
        ("q_factor", q_factor),
        ("alpha_gaia", alpha_gaia),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be finite and positive."));
        }
    }
    if !p_percolation.is_finite() || !(0.0..1.0).contains(&p_percolation) {
        return Err("p_percolation must be finite and in (0, 1).".to_string());
    }
    Ok(())
}

fn validate_kernel_inputs(
    phi: &[f64],
    sync_inputs: &[f64],
    alpha: f64,
    freq: f64,
    q_factor: f64,
    p_percolation: f64,
    t: f64,
    dt: f64,
) -> Result<(), String> {
    if phi.is_empty() || phi.len() != sync_inputs.len() {
        return Err("phi and sync input dimensions must be equal and non-empty.".to_string());
    }
    if phi.iter().any(|value| !value.is_finite())
        || sync_inputs.iter().any(|value| !value.is_finite())
    {
        return Err("phi and sync inputs must be finite.".to_string());
    }
    validate_params(phi.len(), 1, freq, q_factor, alpha, p_percolation)?;
    if !t.is_finite() {
        return Err("time must be finite.".to_string());
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and positive.".to_string());
    }
    Ok(())
}

fn project_inputs(
    inputs: Option<&[Vec<f64>]>,
    n_regions: usize,
    bitstream_length: usize,
) -> Result<Vec<f64>, String> {
    let Some(rows) = inputs else {
        return Ok(vec![0.0; n_regions]);
    };
    if rows.is_empty() {
        return Err("inputs must contain at least one row.".to_string());
    }

    let mut raw = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != bitstream_length {
            return Err("inputs bitstream_length must match adapter parameters.".to_string());
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err("inputs must contain only finite values.".to_string());
        }
        raw.push(row.iter().sum::<f64>() / row.len() as f64);
    }

    if raw.len() == n_regions {
        return Ok(raw);
    }

    let projected = mean(&raw);
    Ok(vec![projected; n_regions])
}

fn logistic(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_abs(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().map(|value| value.abs()).sum::<f64>() / values.len() as f64
}

fn next_unit_interval(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

pub fn validate_l6_plan(state: &L6_PlanetaryAdapter) -> bool {
    if validate_params(
        state.n_regions,
        state.bitstream_length,
        state.f_schumann,
        state.q_factor,
        state.alpha_gaia,
        state.p_percolation,
    )
    .is_err()
    {
        return false;
    }
    state.phi_planetary.len() == state.n_regions
        && state.regional_coherence.len() == state.n_regions
        && state.phi_planetary.iter().all(|value| value.is_finite())
        && state
            .regional_coherence
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        && state.t.is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l6_plan_new() {
        let state = L6_PlanetaryAdapter::new();
        assert!(validate_l6_plan(&state));
    }

    #[test]
    fn test_l6_plan_rejects_invalid_parameters() {
        assert!(L6_PlanetaryAdapter::try_new(0, 1024, 7.83, 4.0, 0.05, 0.592).is_err());
        assert!(L6_PlanetaryAdapter::try_new(100, 0, 7.83, 4.0, 0.05, 0.592).is_err());
        assert!(L6_PlanetaryAdapter::try_new(100, 1024, 0.0, 4.0, 0.05, 0.592).is_err());
        assert!(L6_PlanetaryAdapter::try_new(100, 1024, 7.83, 0.0, 0.05, 0.592).is_err());
        assert!(L6_PlanetaryAdapter::try_new(100, 1024, 7.83, 4.0, 0.0, 0.592).is_err());
        assert!(L6_PlanetaryAdapter::try_new(100, 1024, 7.83, 4.0, 0.05, 1.0).is_err());
    }

    #[test]
    fn test_l6_plan_q_factor_amplifies_coherent_drive() {
        let mut low_q = L6_PlanetaryAdapter::try_new(8, 16, 7.83, 1.0, 0.05, 0.592).unwrap();
        let mut high_q = L6_PlanetaryAdapter::try_new(8, 16, 7.83, 8.0, 0.05, 0.592).unwrap();
        let inputs = vec![vec![1.0; 16]; 8];

        low_q.step_jax(0.01, Some(&inputs)).unwrap();
        high_q.step_jax(0.01, Some(&inputs)).unwrap();

        assert!(mean_abs(&high_q.phi_planetary) > mean_abs(&low_q.phi_planetary));
    }

    #[test]
    fn test_l6_plan_percolation_threshold_controls_regional_coherence() {
        let mut low_threshold = L6_PlanetaryAdapter::try_new(8, 16, 7.83, 4.0, 0.05, 0.2).unwrap();
        let mut high_threshold = L6_PlanetaryAdapter::try_new(8, 16, 7.83, 4.0, 0.05, 0.8).unwrap();
        let inputs = vec![vec![0.5; 16]; 8];

        low_threshold.step_jax(0.01, Some(&inputs)).unwrap();
        high_threshold.step_jax(0.01, Some(&inputs)).unwrap();

        assert!(mean(&low_threshold.regional_coherence) > mean(&high_threshold.regional_coherence));
    }

    #[test]
    fn test_l6_plan_step_returns_encoded_regional_bitstreams() {
        let mut state = L6_PlanetaryAdapter::try_new(6, 12, 7.83, 4.0, 0.05, 0.592).unwrap();
        let encoded = state.step_jax(0.01, None).unwrap();

        assert_eq!(encoded.len(), 6);
        assert_eq!(encoded[0].len(), 12);
        assert!(state.phi_planetary.iter().all(|value| value.is_finite()));
        assert!(state
            .regional_coherence
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn test_l6_plan_broadcasts_mismatched_input_regions() {
        let mut state = L6_PlanetaryAdapter::try_new(4, 6, 7.83, 4.0, 0.05, 0.592).unwrap();
        let inputs = vec![vec![0.0; 6], vec![1.0; 6]];

        let encoded = state.step_jax(0.05, Some(&inputs)).unwrap();

        assert_eq!(encoded.len(), 4);
        assert!(state
            .phi_planetary
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() < 1.0e-12));
    }

    #[test]
    fn test_l6_plan_rejects_malformed_inputs_without_mutation() {
        let mut state = L6_PlanetaryAdapter::try_new(3, 4, 7.83, 4.0, 0.05, 0.592).unwrap();
        let before_phi = state.phi_planetary.clone();
        let before_coherence = state.regional_coherence.clone();

        assert!(state.step_jax(0.0, Some(&vec![vec![1.0; 4]])).is_err());
        assert!(state.step_jax(0.05, Some(&Vec::<Vec<f64>>::new())).is_err());
        assert!(state.step_jax(0.05, Some(&vec![vec![1.0; 3]])).is_err());
        assert!(state
            .step_jax(0.05, Some(&vec![vec![f64::NAN; 4]]))
            .is_err());

        assert_eq!(state.t, 0.0);
        assert_eq!(state.phi_planetary, before_phi);
        assert_eq!(state.regional_coherence, before_coherence);
    }
}
