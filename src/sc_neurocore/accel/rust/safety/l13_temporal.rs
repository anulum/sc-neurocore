// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for l13_temporal

#![allow(non_camel_case_types, non_snake_case)]

const DEFAULT_RNG_SEED: u64 = 0x4c31_335f_5449_4d45;
const VALID_TERMINALS: [&str; 7] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"];
const SOURCE_TERMINALS: [&str; 2] = ["T5", "T6"];

#[derive(Debug, Clone)]
pub struct L12SourceInput {
    pub coherence: Vec<f64>,
    pub gaian_stabilization_drive: f64,
    pub noospheric_entropy_load: f64,
    pub effective_dephasing_gamma: f64,
    pub boundary_context_id: Option<String>,
    pub boundary_terminals: Vec<String>,
}

impl L12SourceInput {
    pub fn from_coherence(coherence: Vec<f64>) -> Self {
        Self {
            coherence,
            gaian_stabilization_drive: 0.0,
            noospheric_entropy_load: 0.0,
            effective_dephasing_gamma: 0.0,
            boundary_context_id: None,
            boundary_terminals: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct L13StepOutput {
    pub binding_matrix: Vec<Vec<f64>>,
    pub binding_strength: f64,
    pub source_sampling_signal: Vec<f64>,
    pub source_sampling_gain: f64,
    pub temporal_decoherence_load: f64,
    pub boundary_context_id: Option<String>,
    pub boundary_terminals: Vec<String>,
    pub source_terminal_set: Vec<String>,
    pub source_sampling_bandwidth: f64,
    pub output_bitstreams: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct L13_TemporalLayer {
    pub n_channels: usize,
    pub bitstream_length: usize,
    pub binding_window: usize,
    pub binding_threshold: f64,
    pub quantum_info_coupling: f64,
    pub source_decoherence_coupling: f64,
    pub history: Vec<Vec<f64>>,
    pub binding_matrix: Vec<Vec<f64>>,
    pub step_count: usize,
    pub time: f64,
    pub rng_state: u64,
}

#[derive(Debug, Clone)]
struct SourceContext {
    boundary_context_id: Option<String>,
    boundary_terminals: Vec<String>,
    source_terminal_set: Vec<String>,
    source_sampling_bandwidth: f64,
}

impl L13_TemporalLayer {
    pub fn new() -> Self {
        Self::try_new(64, 1024, 10, 0.5, 0.1, 0.1, None)
            .expect("default L13 temporal parameters are valid")
    }

    pub fn try_new(
        n_channels: usize,
        bitstream_length: usize,
        binding_window: usize,
        binding_threshold: f64,
        quantum_info_coupling: f64,
        source_decoherence_coupling: f64,
        rng_seed: Option<u64>,
    ) -> Result<Self, String> {
        validate_constructor_params(
            n_channels,
            bitstream_length,
            binding_window,
            binding_threshold,
            quantum_info_coupling,
            source_decoherence_coupling,
        )?;

        Ok(Self {
            n_channels,
            bitstream_length,
            binding_window,
            binding_threshold,
            quantum_info_coupling,
            source_decoherence_coupling,
            history: vec![vec![0.0; binding_window]; n_channels],
            binding_matrix: vec![vec![0.0; n_channels]; n_channels],
            step_count: 0,
            time: 0.0,
            rng_state: rng_seed.unwrap_or(DEFAULT_RNG_SEED),
        })
    }

    pub fn step(
        &mut self,
        dt: f64,
        l12_input: Option<&L12SourceInput>,
    ) -> Result<L13StepOutput, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and positive".to_string());
        }

        self.time += dt;
        self.step_count += 1;

        let mut signal = vec![0.0; self.n_channels];
        let mut source_sampling_gain = 0.0;
        let mut temporal_decoherence_load = 0.0;
        let mut source_context = SourceContext::default_context();

        if let Some(input) = l12_input {
            signal = coherence_signal(&input.coherence, self.n_channels)?;
            source_context = source_context_from_input(input)?;
            source_sampling_gain = self.quantum_info_coupling
                * finite_scalar(
                    input.gaian_stabilization_drive,
                    "gaian_stabilization_drive",
                    None,
                )?
                * source_context.source_sampling_bandwidth;
            temporal_decoherence_load = finite_scalar(
                input.noospheric_entropy_load,
                "noospheric_entropy_load",
                Some(0.0),
            )? + finite_scalar(
                input.effective_dephasing_gamma,
                "effective_dephasing_gamma",
                Some(0.0),
            )?;

            let decoherence_penalty = self.source_decoherence_coupling * temporal_decoherence_load;
            for value in &mut signal {
                *value = clamp01(*value + source_sampling_gain - decoherence_penalty);
            }
        }

        for (channel, value) in signal.iter().enumerate() {
            self.history[channel].rotate_left(1);
            self.history[channel][self.binding_window - 1] = *value;
        }

        if self.step_count >= self.binding_window {
            self.binding_matrix = max_lag_binding_matrix(&self.history, self.binding_window);
        }

        let binding_strength = self.binding_strength();
        let output_bitstreams = self.emit_bitstreams();

        Ok(L13StepOutput {
            binding_matrix: self.binding_matrix.clone(),
            binding_strength,
            source_sampling_signal: signal,
            source_sampling_gain,
            temporal_decoherence_load,
            boundary_context_id: source_context.boundary_context_id,
            boundary_terminals: source_context.boundary_terminals,
            source_terminal_set: source_context.source_terminal_set,
            source_sampling_bandwidth: source_context.source_sampling_bandwidth,
            output_bitstreams,
        })
    }

    pub fn get_global_metric(&self) -> f64 {
        if self.n_channels <= 1 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for i in 0..self.n_channels {
            for j in 0..self.n_channels {
                if i != j {
                    total += self.binding_matrix[i][j].abs();
                    count += 1;
                }
            }
        }
        total / count as f64
    }

    fn binding_strength(&self) -> f64 {
        if self.n_channels <= 1 {
            return 0.0;
        }
        let mut bound_pairs = 0usize;
        let mut off_diagonal = 0usize;
        for i in 0..self.n_channels {
            for j in 0..self.n_channels {
                if i != j {
                    off_diagonal += 1;
                    if self.binding_matrix[i][j].abs() > self.binding_threshold {
                        bound_pairs += 1;
                    }
                }
            }
        }
        bound_pairs as f64 / off_diagonal as f64
    }

    fn emit_bitstreams(&mut self) -> Vec<Vec<u8>> {
        let mut bitstreams = vec![vec![0u8; self.bitstream_length]; self.n_channels];
        for (channel, channel_bits) in bitstreams.iter_mut().enumerate() {
            let activation = clamp01(self.binding_matrix[channel][channel] * 0.5 + 0.5);
            for bit in channel_bits {
                *bit = u8::from(self.next_unit_interval() < activation);
            }
        }
        bitstreams
    }

    fn next_unit_interval(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng_state >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

impl SourceContext {
    fn default_context() -> Self {
        Self {
            boundary_context_id: None,
            boundary_terminals: Vec::new(),
            source_terminal_set: Vec::new(),
            source_sampling_bandwidth: 1.0,
        }
    }
}

fn validate_constructor_params(
    n_channels: usize,
    bitstream_length: usize,
    binding_window: usize,
    binding_threshold: f64,
    quantum_info_coupling: f64,
    source_decoherence_coupling: f64,
) -> Result<(), String> {
    if n_channels == 0 {
        return Err("n_channels must be positive".to_string());
    }
    if bitstream_length == 0 {
        return Err("bitstream_length must be positive".to_string());
    }
    if binding_window <= 1 {
        return Err("binding_window must be greater than one".to_string());
    }
    if !binding_threshold.is_finite() || !(0.0..=1.0).contains(&binding_threshold) {
        return Err("binding_threshold must be in [0, 1]".to_string());
    }
    if !quantum_info_coupling.is_finite() || quantum_info_coupling < 0.0 {
        return Err("quantum_info_coupling must be finite and non-negative".to_string());
    }
    if !source_decoherence_coupling.is_finite() || source_decoherence_coupling < 0.0 {
        return Err("source_decoherence_coupling must be finite and non-negative".to_string());
    }
    Ok(())
}

fn coherence_signal(coherence: &[f64], n_channels: usize) -> Result<Vec<f64>, String> {
    if coherence.is_empty() {
        return Err("coherence must contain at least one value".to_string());
    }
    let mut signal = vec![0.0; n_channels];
    for (idx, value) in coherence.iter().take(n_channels).enumerate() {
        if !value.is_finite() {
            return Err("coherence must contain only finite values".to_string());
        }
        if !(0.0..=1.0).contains(value) {
            return Err("coherence values must be within [0, 1]".to_string());
        }
        signal[idx] = *value;
    }
    for value in coherence.iter().skip(n_channels) {
        if !value.is_finite() {
            return Err("coherence must contain only finite values".to_string());
        }
        if !(0.0..=1.0).contains(value) {
            return Err("coherence values must be within [0, 1]".to_string());
        }
    }
    Ok(signal)
}

fn source_context_from_input(input: &L12SourceInput) -> Result<SourceContext, String> {
    match (
        input.boundary_context_id.as_ref(),
        input.boundary_terminals.is_empty(),
    ) {
        (None, true) => Ok(SourceContext::default_context()),
        (None, false) => {
            Err("boundary context requires boundary_context_id and boundary_terminals".to_string())
        }
        (Some(context_id), true) if context_id.is_empty() => {
            Err("boundary_context_id must be non-empty".to_string())
        }
        (Some(_), true) => {
            Err("boundary context requires boundary_context_id and boundary_terminals".to_string())
        }
        (Some(context_id), false) => {
            if context_id.is_empty() {
                return Err("boundary_context_id must be non-empty".to_string());
            }
            for terminal in &input.boundary_terminals {
                if !VALID_TERMINALS.contains(&terminal.as_str()) {
                    return Err(
                        "boundary_terminals must contain valid T1-T7 terminal identifiers"
                            .to_string(),
                    );
                }
            }
            let source_terminal_set = input
                .boundary_terminals
                .iter()
                .filter(|terminal| SOURCE_TERMINALS.contains(&terminal.as_str()))
                .cloned()
                .collect::<Vec<_>>();
            Ok(SourceContext {
                boundary_context_id: Some(context_id.clone()),
                boundary_terminals: input.boundary_terminals.clone(),
                source_sampling_bandwidth: source_terminal_set.len() as f64 / 2.0,
                source_terminal_set,
            })
        }
    }
}

fn finite_scalar(value: f64, name: &str, lower_bound: Option<f64>) -> Result<f64, String> {
    if !value.is_finite() {
        return Err(format!("{name} must be a finite scalar"));
    }
    if lower_bound.is_some_and(|lower| value < lower) {
        return Err(format!("{name} must be finite and non-negative"));
    }
    Ok(value)
}

fn max_lag_binding_matrix(history: &[Vec<f64>], binding_window: usize) -> Vec<Vec<f64>> {
    let n = history.len();
    let window = history.first().map_or(0, Vec::len);
    let max_lag = binding_window
        .saturating_sub(1)
        .min(window.saturating_sub(1)) as isize;
    let mut matrix = vec![vec![0.0; n]; n];
    for (idx, row) in matrix.iter_mut().enumerate() {
        row[idx] = 1.0;
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let mut best = 0.0;
            for lag in -max_lag..=max_lag {
                let corr = lagged_pearson(&history[i], &history[j], lag);
                if corr.abs() > f64::abs(best) {
                    best = corr;
                }
            }
            matrix[i][j] = best;
            matrix[j][i] = best;
        }
    }
    matrix
}

fn lagged_pearson(a: &[f64], b: &[f64], lag: isize) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }
    let window = a.len();
    match lag.cmp(&0) {
        std::cmp::Ordering::Less => {
            let shift = (-lag) as usize;
            pearson(&a[..window - shift], &b[shift..])
        }
        std::cmp::Ordering::Greater => {
            let shift = lag as usize;
            pearson(&a[shift..], &b[..window - shift])
        }
        std::cmp::Ordering::Equal => pearson(a, b),
    }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (a_value, b_value) in a.iter().zip(b) {
        let centered_a = *a_value - mean_a;
        let centered_b = *b_value - mean_b;
        dot += centered_a * centered_b;
        norm_a += centered_a * centered_a;
        norm_b += centered_b * centered_b;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

pub fn validate_l13_temporal(state: &L13_TemporalLayer) -> bool {
    if validate_constructor_params(
        state.n_channels,
        state.bitstream_length,
        state.binding_window,
        state.binding_threshold,
        state.quantum_info_coupling,
        state.source_decoherence_coupling,
    )
    .is_err()
    {
        return false;
    }
    if !state.time.is_finite() {
        return false;
    }
    if state.history.len() != state.n_channels
        || state.binding_matrix.len() != state.n_channels
        || state
            .history
            .iter()
            .any(|row| row.len() != state.binding_window)
        || state
            .binding_matrix
            .iter()
            .any(|row| row.len() != state.n_channels)
    {
        return false;
    }
    state
        .history
        .iter()
        .flatten()
        .chain(state.binding_matrix.iter().flatten())
        .all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l13_temporal_new() {
        let state = L13_TemporalLayer::new();
        assert!(validate_l13_temporal(&state));
        assert_eq!(state.n_channels, 64);
        assert_eq!(state.bitstream_length, 1024);
        assert_eq!(state.binding_window, 10);
    }

    #[test]
    fn test_l13_temporal_step_emits_full_bitstream_surface() {
        let mut state = L13_TemporalLayer::new();
        let output = state.step(0.001, None).unwrap();
        assert_eq!(output.output_bitstreams.len(), 64);
        assert_eq!(output.output_bitstreams[0].len(), 1024);
        assert_eq!(output.source_sampling_signal, vec![0.0; 64]);
        assert!(validate_l13_temporal(&state));
    }

    #[test]
    fn test_l13_temporal_binding_uses_lagged_correlation() {
        let mut layer = L13_TemporalLayer::try_new(2, 16, 6, 0.7, 0.1, 0.1, Some(123)).unwrap();
        let inputs = [
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];

        let mut output = L13StepOutput::default();
        for coherence in inputs {
            output = layer
                .step(0.001, Some(&L12SourceInput::from_coherence(coherence)))
                .unwrap();
        }

        assert!(output.binding_matrix[0][1].abs() > 0.9);
        assert!((output.binding_strength - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_l13_source_sampling_uses_l12_gaian_and_boundary_terms() {
        let mut layer = L13_TemporalLayer::try_new(3, 16, 3, 0.5, 0.5, 0.0, Some(44)).unwrap();
        let input = L12SourceInput {
            coherence: vec![0.2, 0.2, 0.2],
            gaian_stabilization_drive: 0.4,
            noospheric_entropy_load: 0.0,
            effective_dephasing_gamma: 0.0,
            boundary_context_id: Some("ebs-l13".to_string()),
            boundary_terminals: vec!["T5".to_string(), "T6".to_string()],
        };

        let output = layer.step(0.5, Some(&input)).unwrap();

        assert_eq!(
            output.source_terminal_set,
            vec!["T5".to_string(), "T6".to_string()]
        );
        assert!((output.source_sampling_bandwidth - 1.0).abs() < 1.0e-12);
        assert!((output.source_sampling_gain - 0.2).abs() < 1.0e-12);
        assert!(output
            .source_sampling_signal
            .iter()
            .all(|value| (*value - 0.4).abs() < 1.0e-12));
    }

    #[test]
    fn test_l13_temporal_rejects_invalid_parameters_and_inputs() {
        assert!(L13_TemporalLayer::try_new(0, 16, 3, 0.5, 0.1, 0.1, None).is_err());
        assert!(L13_TemporalLayer::try_new(3, 0, 3, 0.5, 0.1, 0.1, None).is_err());
        assert!(L13_TemporalLayer::try_new(3, 16, 1, 0.5, 0.1, 0.1, None).is_err());
        assert!(L13_TemporalLayer::try_new(3, 16, 3, 1.2, 0.1, 0.1, None).is_err());
        assert!(L13_TemporalLayer::try_new(3, 16, 3, 0.5, -0.1, 0.1, None).is_err());

        let mut layer = L13_TemporalLayer::new();
        assert!(layer.step(0.0, None).is_err());
        assert!(layer
            .step(0.001, Some(&L12SourceInput::from_coherence(vec![f64::NAN])))
            .is_err());
    }

    #[test]
    fn test_l13_temporal_rejects_incomplete_boundary_contract() {
        let mut layer = L13_TemporalLayer::try_new(3, 16, 3, 0.5, 0.5, 0.0, Some(44)).unwrap();
        let invalid = L12SourceInput {
            coherence: vec![0.2, 0.2, 0.2],
            gaian_stabilization_drive: 0.4,
            noospheric_entropy_load: 0.0,
            effective_dephasing_gamma: 0.0,
            boundary_context_id: None,
            boundary_terminals: vec!["T5".to_string()],
        };

        assert!(layer.step(0.001, Some(&invalid)).is_err());
    }
}
