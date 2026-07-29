// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for L7 Metatron routing

#![allow(non_camel_case_types, non_snake_case)]

#[derive(Debug, Clone)]
pub struct L7_SymbolicAdapter {
    pub n_nodes: usize,
    pub bitstream_length: usize,
    pub g_geometric_gain: f64,
    pub phi_golden_ratio: f64,
    pub coupling_leak: f64,
    pub rng_state: u64,
    pub node_phases: Vec<f64>,
    pub metatron_matrix: Vec<Vec<f64>>,
}

impl L7_SymbolicAdapter {
    pub fn new() -> Self {
        Self::try_new(13, 1024, 1.2, 1.618_033_988_75, 0.05)
            .expect("default L7 symbolic adapter parameters are valid")
    }

    pub fn try_new(
        n_nodes: usize,
        bitstream_length: usize,
        g_geometric_gain: f64,
        phi_golden_ratio: f64,
        coupling_leak: f64,
    ) -> Result<Self, String> {
        validate_params(
            n_nodes,
            bitstream_length,
            g_geometric_gain,
            phi_golden_ratio,
            coupling_leak,
        )?;

        let metatron_matrix =
            init_metatron_matrix(n_nodes, g_geometric_gain, phi_golden_ratio, coupling_leak)?;

        Ok(Self {
            n_nodes,
            bitstream_length,
            g_geometric_gain,
            phi_golden_ratio,
            coupling_leak,
            rng_state: 0x4d45_5441_5452_4f4e,
            node_phases: vec![0.0; n_nodes],
            metatron_matrix,
        })
    }

    pub fn encode(&mut self) -> Vec<Vec<u8>> {
        let mut out = vec![vec![0_u8; self.bitstream_length]; self.n_nodes];

        for (row, phase) in out.iter_mut().zip(self.node_phases.iter()) {
            let activation = ((1.0 + phase.cos()) * 0.5).clamp(0.0, 1.0);
            for bit in row.iter_mut() {
                *bit = u8::from(next_unit_interval(&mut self.rng_state) < activation);
            }
        }

        out
    }

    pub fn symbolic_kernel(
        phases: &[f64],
        metatron: &[Vec<f64>],
        inputs: &[f64],
        dt: f64,
    ) -> Result<Vec<f64>, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and positive.".to_string());
        }
        if phases.len() != metatron.len() || phases.len() != inputs.len() {
            return Err("phase, routing, and input dimensions must match.".to_string());
        }
        if phases.iter().any(|value| !value.is_finite())
            || inputs.iter().any(|value| !value.is_finite())
        {
            return Err("phases and inputs must be finite.".to_string());
        }

        let mut next = vec![0.0; phases.len()];
        for (row_index, row) in metatron.iter().enumerate() {
            if row.len() != inputs.len() || row.iter().any(|value| !value.is_finite()) {
                return Err("routing matrix must be finite and square.".to_string());
            }
            let drive: f64 = row.iter().zip(inputs.iter()).map(|(w, x)| w * x).sum();
            let d_phase = drive - 0.1 * phases[row_index];
            next[row_index] = phases[row_index] + d_phase * dt;
        }
        Ok(next)
    }

    pub fn step_jax(
        &mut self,
        dt: f64,
        inputs: Option<&[Vec<f64>]>,
    ) -> Result<Vec<Vec<u8>>, String> {
        let input_drive = project_inputs(inputs, self.n_nodes)?;
        self.node_phases =
            Self::symbolic_kernel(&self.node_phases, &self.metatron_matrix, &input_drive, dt)?;
        Ok(self.encode())
    }

    pub fn decode(&self) -> f64 {
        routing_coherence(&self.node_phases)
    }

    pub fn get_metrics(&self) -> (f64, f64) {
        let coherence = routing_coherence(&self.node_phases);
        let stability = self
            .node_phases
            .iter()
            .map(|phase| phase.cos())
            .sum::<f64>()
            / self.node_phases.len() as f64;
        (coherence, stability)
    }
}

fn validate_params(
    n_nodes: usize,
    bitstream_length: usize,
    g_geometric_gain: f64,
    phi_golden_ratio: f64,
    coupling_leak: f64,
) -> Result<(), String> {
    if n_nodes == 0 {
        return Err("n_nodes must be positive.".to_string());
    }
    if bitstream_length == 0 {
        return Err("bitstream_length must be positive.".to_string());
    }
    if !g_geometric_gain.is_finite() || g_geometric_gain <= 0.0 {
        return Err("g_geometric_gain must be finite and positive.".to_string());
    }
    if !phi_golden_ratio.is_finite() || phi_golden_ratio <= 0.0 {
        return Err("phi_golden_ratio must be finite and positive.".to_string());
    }
    if !coupling_leak.is_finite() || !(0.0..1.0).contains(&coupling_leak) {
        return Err("coupling_leak must be finite and in [0, 1).".to_string());
    }
    Ok(())
}

fn init_metatron_matrix(
    n_nodes: usize,
    geometric_gain: f64,
    golden_ratio: f64,
    coupling_leak: f64,
) -> Result<Vec<Vec<f64>>, String> {
    if n_nodes == 1 {
        return Ok(vec![vec![1.0]]);
    }

    let coords = metatron_coordinates(n_nodes);
    let mut off_diag = vec![vec![0.0; n_nodes]; n_nodes];
    for row in 0..n_nodes {
        for col in 0..n_nodes {
            if row == col {
                continue;
            }
            let dx = coords[row].0 - coords[col].0;
            let dy = coords[row].1 - coords[col].1;
            let distance = (dx * dx + dy * dy).sqrt();
            off_diag[row][col] = geometric_gain * (-distance / golden_ratio).exp();
        }
    }

    let max_row_sum = off_diag
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .fold(0.0, f64::max);
    if max_row_sum <= 0.0 {
        return Err("Metatron topology requires at least one off-diagonal edge.".to_string());
    }

    let scale = (1.0 - coupling_leak) / max_row_sum;
    for row in off_diag.iter_mut() {
        for value in row.iter_mut() {
            *value *= scale;
        }
    }

    for (row, values) in off_diag.iter_mut().enumerate() {
        let row_sum = values.iter().sum::<f64>();
        values[row] = 1.0 - row_sum;
    }
    Ok(off_diag)
}

fn metatron_coordinates(n_nodes: usize) -> Vec<(f64, f64)> {
    if n_nodes == 13 {
        let mut coords = Vec::with_capacity(13);
        coords.push((0.0, 0.0));
        for radius in [1.0, 2.0] {
            for index in 0..6 {
                let angle = 2.0 * std::f64::consts::PI * index as f64 / 6.0;
                coords.push((radius * angle.cos(), radius * angle.sin()));
            }
        }
        return coords;
    }

    let mut coords = Vec::with_capacity(n_nodes);
    coords.push((0.0, 0.0));
    for index in 0..(n_nodes - 1) {
        let angle = 2.0 * std::f64::consts::PI * index as f64 / (n_nodes - 1) as f64;
        coords.push((angle.cos(), angle.sin()));
    }
    coords
}

fn project_inputs(inputs: Option<&[Vec<f64>]>, n_nodes: usize) -> Result<Vec<f64>, String> {
    let Some(rows) = inputs else {
        return Ok(vec![0.0; n_nodes]);
    };
    if rows.is_empty() {
        return Ok(vec![0.0; n_nodes]);
    }

    let mut raw = Vec::with_capacity(rows.len());
    for row in rows {
        if row.is_empty() {
            return Err("input rows must not be empty.".to_string());
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err("inputs must contain only finite values.".to_string());
        }
        raw.push(row.iter().sum::<f64>() / row.len() as f64);
    }

    if raw.len() == n_nodes {
        return Ok(raw);
    }

    let mean = raw.iter().sum::<f64>() / raw.len() as f64;
    Ok(vec![mean; n_nodes])
}

fn routing_coherence(phases: &[f64]) -> f64 {
    if phases.is_empty() {
        return 0.0;
    }
    let real = phases.iter().map(|phase| phase.cos()).sum::<f64>() / phases.len() as f64;
    let imag = phases.iter().map(|phase| phase.sin()).sum::<f64>() / phases.len() as f64;
    (real * real + imag * imag).sqrt()
}

fn next_unit_interval(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

pub fn validate_l7_sym(state: &L7_SymbolicAdapter) -> bool {
    if validate_params(
        state.n_nodes,
        state.bitstream_length,
        state.g_geometric_gain,
        state.phi_golden_ratio,
        state.coupling_leak,
    )
    .is_err()
    {
        return false;
    }
    if state.node_phases.len() != state.n_nodes || state.metatron_matrix.len() != state.n_nodes {
        return false;
    }
    state.metatron_matrix.iter().all(|row| {
        row.len() == state.n_nodes
            && row.iter().all(|value| value.is_finite() && *value >= 0.0)
            && (row.iter().sum::<f64>() - 1.0).abs() < 1.0e-10
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l7_sym_new() {
        let state = L7_SymbolicAdapter::new();
        assert!(validate_l7_sym(&state));
    }

    #[test]
    fn test_l7_sym_metatron_matrix_is_row_stochastic_geometry() {
        let state = L7_SymbolicAdapter::new();
        assert_eq!(state.metatron_matrix.len(), 13);
        assert_eq!(state.metatron_matrix[0].len(), 13);

        for row in &state.metatron_matrix {
            let row_sum: f64 = row.iter().sum();
            assert!((row_sum - 1.0).abs() < 1.0e-12);
        }

        assert!((state.metatron_matrix[0][0] - state.coupling_leak).abs() < 1.0e-12);
        assert!(state.metatron_matrix[0][1] > 0.0);
        assert!((state.metatron_matrix[2][8] - state.metatron_matrix[8][2]).abs() < 1.0e-12);
    }

    #[test]
    fn test_l7_sym_rejects_invalid_parameters() {
        assert!(L7_SymbolicAdapter::try_new(0, 1024, 1.2, 1.61803398875, 0.05).is_err());
        assert!(L7_SymbolicAdapter::try_new(13, 0, 1.2, 1.61803398875, 0.05).is_err());
        assert!(L7_SymbolicAdapter::try_new(13, 1024, 0.0, 1.61803398875, 0.05).is_err());
        assert!(L7_SymbolicAdapter::try_new(13, 1024, 1.2, 0.0, 0.05).is_err());
        assert!(L7_SymbolicAdapter::try_new(13, 1024, 1.2, 1.61803398875, 1.0).is_err());
    }

    #[test]
    fn test_l7_sym_step_uses_routing_matrix() {
        let mut state = L7_SymbolicAdapter::new();
        let inputs = vec![vec![1.0; 16]; 13];
        let encoded = state.step_jax(0.1, Some(&inputs)).unwrap();

        assert_eq!(encoded.len(), 13);
        assert_eq!(encoded[0].len(), 1024);
        assert!(state.node_phases.iter().all(|phase| phase.is_finite()));
        assert!(state.node_phases.iter().any(|phase| *phase > 0.0));
    }
}
