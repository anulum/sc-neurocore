// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety for l7_symbolic

#![allow(non_camel_case_types, non_snake_case)]

const PHI: f64 = 1.618_033_988_749_895;
const DEFAULT_RNG_SEED: u64 = 0x4c37_5359_4d42_4f4c;
const FIBONACCI: [f64; 12] = [
    1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0,
];
const PLATONIC_VERTICES: [usize; 5] = [4, 8, 6, 20, 12];

#[derive(Debug, Clone)]
pub struct L7SymbolicParams {
    pub n_symbols: usize,
    pub n_meridians: usize,
    pub n_acupoints: usize,
    pub bitstream_length: usize,
    pub phi_alignment_weight: f64,
    pub fibonacci_weight: f64,
    pub metatron_weight: f64,
    pub platonic_weight: f64,
    pub e8_weight: f64,
    pub symbol_decay: f64,
    pub symbol_coupling: f64,
    pub glyph_dimensions: usize,
    pub ecological_coupling: f64,
    pub cosmic_coupling: f64,
    pub rng_seed: Option<u64>,
}

impl Default for L7SymbolicParams {
    fn default() -> Self {
        Self {
            n_symbols: 128,
            n_meridians: 12,
            n_acupoints: 361,
            bitstream_length: 1024,
            phi_alignment_weight: 0.25,
            fibonacci_weight: 0.2,
            metatron_weight: 0.2,
            platonic_weight: 0.15,
            e8_weight: 0.2,
            symbol_decay: 0.05,
            symbol_coupling: 0.3,
            glyph_dimensions: 6,
            ecological_coupling: 0.1,
            cosmic_coupling: 0.15,
            rng_seed: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct L6SymbolicInput {
    pub symbolic_drive: Option<Vec<f64>>,
    pub schumann_field: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Copy)]
pub struct AcupointStimulus {
    pub point_id: usize,
    pub intensity: f64,
}

#[derive(Debug, Clone, Default)]
pub struct L7StepOutput {
    pub glyph_vector: Vec<f64>,
    pub phi_alignment: f64,
    pub fibonacci_alignment: f64,
    pub metatron_flow: f64,
    pub platonic_coherence: f64,
    pub e8_alignment: f64,
    pub symbolic_health: f64,
    pub cosmic_phase_drive: f64,
    pub meridian_qi: Vec<f64>,
    pub acupoint_activations: Vec<f64>,
    pub e8_state: Vec<f64>,
    pub output_bitstreams: Vec<Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct L7_SymbolicLayer {
    pub params: L7SymbolicParams,
    pub symbol_activations: Vec<f64>,
    pub phi_alignment: f64,
    pub fibonacci_alignment: f64,
    pub metatron_flow: f64,
    pub platonic_coherence: f64,
    pub e8_alignment: f64,
    pub symbolic_health: f64,
    pub meridian_qi: Vec<f64>,
    pub acupoint_activations: Vec<f64>,
    pub glyph_vector: Vec<f64>,
    pub e8_state: Vec<f64>,
    pub time: f64,
    pub rng_state: u64,
}

impl L7_SymbolicLayer {
    pub fn new() -> Self {
        Self::try_new(L7SymbolicParams::default()).expect("default L7 parameters are valid")
    }

    pub fn try_new(params: L7SymbolicParams) -> Result<Self, String> {
        validate_params(&params)?;
        let mut layer = Self {
            rng_state: params.rng_seed.unwrap_or(DEFAULT_RNG_SEED),
            symbol_activations: vec![0.0; params.n_symbols],
            phi_alignment: 0.5,
            fibonacci_alignment: 0.5,
            metatron_flow: 0.5,
            platonic_coherence: 0.5,
            e8_alignment: 0.5,
            symbolic_health: 0.5,
            meridian_qi: vec![0.5; params.n_meridians],
            acupoint_activations: vec![0.0; params.n_acupoints],
            glyph_vector: vec![0.0; params.glyph_dimensions],
            e8_state: vec![0.0; 8],
            time: 0.0,
            params,
        };
        for idx in 0..layer.symbol_activations.len() {
            layer.symbol_activations[idx] = layer.next_unit_interval() * 0.3;
        }
        for idx in 0..8 {
            layer.e8_state[idx] = layer.next_unit_interval() * 0.5;
        }
        Ok(layer)
    }

    pub fn step(
        &mut self,
        dt: f64,
        l6_input: Option<&L6SymbolicInput>,
        symbol_input: Option<&[f64]>,
        acupoint_stimulus: Option<&[AcupointStimulus]>,
    ) -> Result<L7StepOutput, String> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err("dt must be finite and positive".to_string());
        }
        self.time += dt;

        if let Some(values) = symbol_input {
            let symbol_values = validate_symbol_input(values, self.params.n_symbols)?;
            for (state, input) in self.symbol_activations.iter_mut().zip(symbol_values) {
                *state = clamp01(*state + input * 0.2);
            }
        }

        self.update_geometry_metrics();
        self.e8_alignment = e8_alignment(&self.e8_state);
        for idx in 0..8 {
            self.e8_state[idx] =
                (self.e8_state[idx] + 0.1 * self.next_normal() * dt).clamp(-1.0, 1.0);
        }

        self.symbolic_health = self.params.phi_alignment_weight * self.phi_alignment
            + self.params.fibonacci_weight * self.fibonacci_alignment
            + self.params.metatron_weight * self.metatron_flow
            + self.params.platonic_weight * self.platonic_coherence
            + self.params.e8_weight * self.e8_alignment;

        self.update_meridians(dt, l6_input)?;
        self.apply_acupoints(dt, acupoint_stimulus)?;
        self.glyph_vector = vec![
            self.phi_alignment,
            self.fibonacci_alignment,
            self.metatron_flow,
            self.platonic_coherence,
            self.e8_alignment,
            self.symbolic_health,
        ];

        self.update_symbols(dt);
        let output_bitstreams = self.emit_bitstreams();

        Ok(L7StepOutput {
            glyph_vector: self.glyph_vector.clone(),
            phi_alignment: self.phi_alignment,
            fibonacci_alignment: self.fibonacci_alignment,
            metatron_flow: self.metatron_flow,
            platonic_coherence: self.platonic_coherence,
            e8_alignment: self.e8_alignment,
            symbolic_health: self.symbolic_health,
            cosmic_phase_drive: self.params.cosmic_coupling * self.symbolic_health,
            meridian_qi: self.meridian_qi.clone(),
            acupoint_activations: self.acupoint_activations.clone(),
            e8_state: self.e8_state.clone(),
            output_bitstreams,
        })
    }

    pub fn get_global_metric(&self) -> f64 {
        self.symbolic_health
    }

    pub fn get_glyph_vector_normalized(&self) -> Vec<f64> {
        let max_value = self
            .glyph_vector
            .iter()
            .copied()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let denom = max_value + 1.0e-8;
        self.glyph_vector
            .iter()
            .map(|value| *value / denom)
            .collect()
    }

    pub fn stimulate_meridian(&mut self, meridian_id: usize, intensity: f64) -> Result<(), String> {
        if meridian_id >= self.params.n_meridians {
            return Err("meridian_id must be in range".to_string());
        }
        if !intensity.is_finite() {
            return Err("intensity must be finite".to_string());
        }
        self.meridian_qi[meridian_id] = clamp01(self.meridian_qi[meridian_id] + intensity);
        Ok(())
    }

    pub fn get_acupoint_map(&self) -> Vec<(String, f64)> {
        [
            ("LI4_Hegu", 4usize),
            ("ST36_Zusanli", 36),
            ("SP6_Sanyinjiao", 60),
            ("PC6_Neiguan", 96),
            ("LV3_Taichong", 120),
            ("GV20_Baihui", 200),
            ("CV4_Guanyuan", 250),
            ("BL23_Shenshu", 300),
        ]
        .into_iter()
        .filter(|(_, idx)| *idx < self.params.n_acupoints)
        .map(|(name, idx)| (name.to_string(), self.acupoint_activations[idx]))
        .collect()
    }

    fn update_geometry_metrics(&mut self) {
        let mut sorted = self.symbol_activations.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        if sorted.len() > 1 && sorted[1] > 0.01 {
            let distances = sorted
                .windows(2)
                .map(|pair| (pair[0] / (pair[1] + 1.0e-8) - PHI).abs())
                .collect::<Vec<_>>();
            self.phi_alignment = (-mean(&distances)).exp();
        } else {
            self.phi_alignment = 0.5;
        }

        let top_8 = sorted.iter().take(8).copied().collect::<Vec<_>>();
        let max_top = top_8.iter().copied().fold(0.0_f64, f64::max);
        if top_8.len() == 8 && max_top > 0.01 {
            let top_norm = top_8
                .iter()
                .map(|value| *value / (max_top + 1.0e-8))
                .collect::<Vec<_>>();
            let fib_norm = FIBONACCI[..8]
                .iter()
                .map(|value| *value / FIBONACCI[7])
                .collect::<Vec<_>>();
            let corr = pearson(&top_norm, &fib_norm);
            self.fibonacci_alignment = ((corr + 1.0) / 2.0).max(0.0);
        } else {
            self.fibonacci_alignment = 0.5;
        }

        let metatron_nodes = self.params.n_symbols.min(13);
        let active_nodes = self
            .symbol_activations
            .iter()
            .take(metatron_nodes)
            .filter(|value| **value > 0.5)
            .count();
        let base_flow = active_nodes as f64 / metatron_nodes as f64;
        self.metatron_flow = 0.9 * base_flow + 0.1 * self.next_unit_interval();

        let metrics = PLATONIC_VERTICES
            .iter()
            .map(|vertices| {
                let count = (*vertices).min(self.symbol_activations.len());
                1.0 - stddev(&self.symbol_activations[..count])
            })
            .collect::<Vec<_>>();
        self.platonic_coherence = mean(&metrics);
    }

    fn update_meridians(
        &mut self,
        dt: f64,
        l6_input: Option<&L6SymbolicInput>,
    ) -> Result<(), String> {
        let old = self.meridian_qi.clone();
        for idx in 0..self.meridian_qi.len() {
            let previous = if idx == 0 {
                old[old.len() - 1]
            } else {
                old[idx - 1]
            };
            self.meridian_qi[idx] += (previous - old[idx]) * self.params.symbol_coupling * dt;
        }

        if let Some(input) = l6_input {
            let effect = l6_symbolic_effect(input)?;
            if effect != 0.0 {
                let gain = 1.0 + self.params.ecological_coupling * effect;
                for value in &mut self.meridian_qi {
                    *value *= gain;
                }
            }
        }
        for value in &mut self.meridian_qi {
            *value = clamp01(*value);
        }
        Ok(())
    }

    fn apply_acupoints(
        &mut self,
        dt: f64,
        stimulus: Option<&[AcupointStimulus]>,
    ) -> Result<(), String> {
        if let Some(stimulus) = stimulus {
            for point in stimulus {
                if point.point_id >= self.params.n_acupoints {
                    return Err("acupoint_stimulus point id out of range".to_string());
                }
                if !point.intensity.is_finite() {
                    return Err("acupoint_stimulus intensities must be finite".to_string());
                }
                self.acupoint_activations[point.point_id] =
                    clamp01(self.acupoint_activations[point.point_id] + point.intensity);
            }
        }

        let decay = (1.0 - self.params.symbol_decay * dt).max(0.0);
        for value in &mut self.acupoint_activations {
            *value *= decay;
        }
        Ok(())
    }

    fn update_symbols(&mut self, dt: f64) {
        let old = self.symbol_activations.clone();
        for idx in 0..self.symbol_activations.len() {
            let left = if idx == 0 {
                old[old.len() - 1]
            } else {
                old[idx - 1]
            };
            let right = if idx + 1 == old.len() {
                old[0]
            } else {
                old[idx + 1]
            };
            self.symbol_activations[idx] +=
                self.params.symbol_coupling * ((left + right) / 2.0 - old[idx]) * dt;
            self.symbol_activations[idx] *= (1.0 - self.params.symbol_decay * dt).max(0.0);
            self.symbol_activations[idx] = clamp01(self.symbol_activations[idx]);
        }
    }

    fn emit_bitstreams(&mut self) -> Vec<Vec<u8>> {
        let mut output_probs = self.symbol_activations.clone();
        output_probs.extend_from_slice(&self.meridian_qi);
        output_probs.extend_from_slice(&self.glyph_vector);
        output_probs.truncate(self.params.n_symbols);

        let mut bitstreams = vec![vec![0u8; self.params.bitstream_length]; self.params.n_symbols];
        for channel in 0..self.params.n_symbols {
            let probability = clamp01(output_probs[channel]);
            for bit in 0..self.params.bitstream_length {
                bitstreams[channel][bit] = u8::from(self.next_unit_interval() < probability);
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

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_unit_interval().max(1.0e-12);
        let u2 = self.next_unit_interval();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn validate_params(params: &L7SymbolicParams) -> Result<(), String> {
    if params.n_symbols < 2 {
        return Err("n_symbols must be at least two".to_string());
    }
    if params.n_meridians == 0 {
        return Err("n_meridians must be positive".to_string());
    }
    if params.n_acupoints == 0 {
        return Err("n_acupoints must be positive".to_string());
    }
    if params.bitstream_length == 0 {
        return Err("bitstream_length must be positive".to_string());
    }
    if params.glyph_dimensions != 6 {
        return Err("glyph_dimensions must be six for the current glyph contract".to_string());
    }
    let weights = [
        params.phi_alignment_weight,
        params.fibonacci_weight,
        params.metatron_weight,
        params.platonic_weight,
        params.e8_weight,
    ];
    if weights
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || weights.iter().sum::<f64>() <= 0.0
    {
        return Err("weights must be finite, non-negative, and sum positive".to_string());
    }
    for (name, value) in [
        ("symbol_decay", params.symbol_decay),
        ("symbol_coupling", params.symbol_coupling),
        ("ecological_coupling", params.ecological_coupling),
        ("cosmic_coupling", params.cosmic_coupling),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} must be finite and non-negative"));
        }
    }
    Ok(())
}

fn validate_symbol_input(values: &[f64], n_symbols: usize) -> Result<Vec<f64>, String> {
    if values.len() < n_symbols {
        return Err("symbol_input must contain at least n_symbols values".to_string());
    }
    let result = values[..n_symbols].to_vec();
    if result.iter().any(|value| !value.is_finite()) {
        return Err("symbol_input must contain only finite values".to_string());
    }
    Ok(result)
}

fn l6_symbolic_effect(input: &L6SymbolicInput) -> Result<f64, String> {
    if let Some(values) = &input.symbolic_drive {
        return unit_mean(values, "symbolic_drive");
    }
    if let Some(values) = &input.schumann_field {
        return finite_mean(values, "schumann_field").map(|value| value - 1.0);
    }
    Ok(0.0)
}

fn finite_mean(values: &[f64], name: &str) -> Result<f64, String> {
    if values.is_empty() {
        return Err(format!("{name} must contain at least one value"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(mean(values))
}

fn unit_mean(values: &[f64], name: &str) -> Result<f64, String> {
    finite_mean(values, name).and_then(|result| {
        if values.iter().any(|value| !(0.0..=1.0).contains(value)) {
            Err(format!("{name} values must be within [0, 1]"))
        } else {
            Ok(result)
        }
    })
}

pub fn e8_roots() -> Vec<Vec<f64>> {
    let mut roots = Vec::with_capacity(240);
    for i in 0..8 {
        for j in (i + 1)..8 {
            for si in [-1.0, 1.0] {
                for sj in [-1.0, 1.0] {
                    let mut root = vec![0.0; 8];
                    root[i] = si;
                    root[j] = sj;
                    roots.push(root);
                }
            }
        }
    }
    for mask in 0..256usize {
        let mut signs = vec![1.0; 8];
        let mut negatives = 0usize;
        for (bit, sign) in signs.iter_mut().enumerate() {
            if ((mask >> bit) & 1) == 1 {
                *sign = -1.0;
                negatives += 1;
            }
        }
        if negatives.is_multiple_of(2) {
            roots.push(signs.into_iter().map(|value| 0.5 * value).collect());
        }
    }
    roots
}

fn e8_alignment(state: &[f64]) -> f64 {
    let norm = l2_norm(state);
    if norm == 0.0 {
        return 0.5;
    }
    let unit = state.iter().map(|value| *value / norm).collect::<Vec<_>>();
    e8_roots()
        .iter()
        .map(|root| {
            let root_norm = l2_norm(root);
            if root_norm == 0.0 {
                0.0
            } else {
                dot(root, &unit).abs() / root_norm
            }
        })
        .fold(0.0_f64, f64::max)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 {
        return 0.0;
    }
    let mean_a = mean(a);
    let mean_b = mean(b);
    let mut numerator = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for (a_value, b_value) in a.iter().zip(b) {
        let da = *a_value - mean_a;
        let db = *b_value - mean_b;
        numerator += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }
    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn stddev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let avg = mean(values);
    (values
        .iter()
        .map(|value| {
            let diff = *value - avg;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a_value, b_value)| a_value * b_value)
        .sum()
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

pub fn validate_l7_symbolic(state: &L7_SymbolicLayer) -> bool {
    validate_params(&state.params).is_ok()
        && state.symbol_activations.len() == state.params.n_symbols
        && state.meridian_qi.len() == state.params.n_meridians
        && state.acupoint_activations.len() == state.params.n_acupoints
        && state.glyph_vector.len() == state.params.glyph_dimensions
        && state.e8_state.len() == 8
        && state.time.is_finite()
        && state
            .symbol_activations
            .iter()
            .chain(&state.meridian_qi)
            .chain(&state.acupoint_activations)
            .chain(&state.glyph_vector)
            .chain(&state.e8_state)
            .all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l7_symbolic_new() {
        let state = L7_SymbolicLayer::new();
        assert!(validate_l7_symbolic(&state));
    }

    #[test]
    fn test_l7_symbolic_step() {
        let mut state = L7_SymbolicLayer::new();
        let output = state.step(0.001, None, None, None).unwrap();
        assert_eq!(output.output_bitstreams.len(), 128);
        assert_eq!(output.output_bitstreams[0].len(), 1024);
        assert!(validate_l7_symbolic(&state));
    }

    #[test]
    fn test_l7_seed_scopes_initial_state_and_output_bitstreams() {
        let params = L7SymbolicParams {
            n_symbols: 16,
            n_meridians: 4,
            n_acupoints: 16,
            bitstream_length: 64,
            rng_seed: Some(123),
            ..L7SymbolicParams::default()
        };
        let mut a = L7_SymbolicLayer::try_new(params.clone()).unwrap();
        let mut b = L7_SymbolicLayer::try_new(params).unwrap();

        assert_eq!(a.symbol_activations, b.symbol_activations);
        assert_eq!(a.e8_state, b.e8_state);
        let a0 = a.step(0.001, None, None, None).unwrap().output_bitstreams;
        let b0 = b.step(0.001, None, None, None).unwrap().output_bitstreams;
        let a1 = a.step(0.001, None, None, None).unwrap().output_bitstreams;
        let b1 = b.step(0.001, None, None, None).unwrap().output_bitstreams;

        assert_eq!(a0, b0);
        assert_eq!(a1, b1);
        assert_ne!(a0, a1);
    }

    #[test]
    fn test_l7_e8_alignment_uses_full_root_system() {
        let roots = e8_roots();
        assert_eq!(roots.len(), 240);
        assert!(roots.iter().all(|root| root.len() == 8));

        let params = L7SymbolicParams {
            n_symbols: 16,
            n_meridians: 4,
            n_acupoints: 16,
            bitstream_length: 16,
            rng_seed: Some(7),
            ..L7SymbolicParams::default()
        };
        let mut layer = L7_SymbolicLayer::try_new(params).unwrap();
        layer.e8_state = vec![0.5; 8];

        let result = layer.step(0.001, None, None, None).unwrap();

        assert!((result.e8_alignment - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_l7_symbolic_health_weights_and_cosmic_drive_are_used() {
        let params = L7SymbolicParams {
            n_symbols: 16,
            n_meridians: 4,
            n_acupoints: 16,
            bitstream_length: 16,
            phi_alignment_weight: 1.0,
            fibonacci_weight: 0.0,
            metatron_weight: 0.0,
            platonic_weight: 0.0,
            e8_weight: 0.0,
            cosmic_coupling: 0.5,
            rng_seed: Some(8),
            ..L7SymbolicParams::default()
        };
        let mut layer = L7_SymbolicLayer::try_new(params).unwrap();
        let result = layer.step(0.001, None, None, None).unwrap();

        assert!((result.symbolic_health - result.phi_alignment).abs() < 1.0e-12);
        assert!((result.cosmic_phase_drive - 0.5 * result.symbolic_health).abs() < 1.0e-12);
    }

    #[test]
    fn test_l7_consumes_l6_symbolic_drive_and_prefers_it_over_schumann() {
        let params = L7SymbolicParams {
            n_symbols: 16,
            n_meridians: 4,
            n_acupoints: 16,
            bitstream_length: 16,
            ecological_coupling: 0.2,
            rng_seed: Some(10),
            ..L7SymbolicParams::default()
        };
        let mut base = L7_SymbolicLayer::try_new(params.clone()).unwrap();
        let mut driven = L7_SymbolicLayer::try_new(params.clone()).unwrap();
        let mut both = L7_SymbolicLayer::try_new(params).unwrap();

        let base_qi = base.step(0.001, None, None, None).unwrap().meridian_qi;
        let driven_qi = driven
            .step(
                0.001,
                Some(&L6SymbolicInput {
                    symbolic_drive: Some(vec![1.0; 8]),
                    schumann_field: None,
                }),
                None,
                None,
            )
            .unwrap()
            .meridian_qi;
        let both_qi = both
            .step(
                0.001,
                Some(&L6SymbolicInput {
                    symbolic_drive: Some(vec![1.0; 8]),
                    schumann_field: Some(vec![0.0; 8]),
                }),
                None,
                None,
            )
            .unwrap()
            .meridian_qi;

        assert!(mean(&driven_qi) > mean(&base_qi));
        assert_eq!(driven_qi, both_qi);
    }

    #[test]
    fn test_l7_rejects_invalid_parameters_and_inputs() {
        assert!(L7_SymbolicLayer::try_new(L7SymbolicParams {
            n_symbols: 1,
            ..L7SymbolicParams::default()
        })
        .is_err());
        assert!(L7_SymbolicLayer::try_new(L7SymbolicParams {
            n_meridians: 0,
            ..L7SymbolicParams::default()
        })
        .is_err());
        assert!(L7_SymbolicLayer::try_new(L7SymbolicParams {
            glyph_dimensions: 5,
            ..L7SymbolicParams::default()
        })
        .is_err());
        assert!(L7_SymbolicLayer::try_new(L7SymbolicParams {
            phi_alignment_weight: f64::NAN,
            ..L7SymbolicParams::default()
        })
        .is_err());

        let mut layer = L7_SymbolicLayer::try_new(L7SymbolicParams {
            n_symbols: 16,
            n_meridians: 4,
            n_acupoints: 16,
            rng_seed: Some(9),
            ..L7SymbolicParams::default()
        })
        .unwrap();
        assert!(layer.step(0.0, None, None, None).is_err());
        assert!(layer
            .step(0.001, None, Some(&[1.0, f64::NAN]), None)
            .is_err());
        assert!(layer.step(0.001, None, Some(&[1.0; 15]), None).is_err());
        assert!(layer
            .step(
                0.001,
                Some(&L6SymbolicInput {
                    symbolic_drive: Some(vec![0.5, f64::NAN]),
                    schumann_field: None,
                }),
                None,
                None,
            )
            .is_err());
        assert!(layer
            .step(
                0.001,
                None,
                None,
                Some(&[AcupointStimulus {
                    point_id: 16,
                    intensity: 0.5,
                }]),
            )
            .is_err());
    }

    #[test]
    fn test_l7_stimulates_meridian_and_named_acupoint_map() {
        let mut layer = L7_SymbolicLayer::try_new(L7SymbolicParams {
            n_symbols: 16,
            n_meridians: 4,
            n_acupoints: 301,
            rng_seed: Some(5),
            ..L7SymbolicParams::default()
        })
        .unwrap();

        layer.stimulate_meridian(0, 0.4).unwrap();
        assert!(layer.meridian_qi[0] > 0.5);
        layer
            .step(
                0.001,
                None,
                None,
                Some(&[AcupointStimulus {
                    point_id: 4,
                    intensity: 0.8,
                }]),
            )
            .unwrap();

        let acupoints = layer.get_acupoint_map();
        assert!(acupoints
            .iter()
            .any(|(name, value)| name == "LI4_Hegu" && *value > 0.0));
    }
}
