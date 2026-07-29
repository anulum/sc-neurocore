// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for transfer checkpoints

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct SNNCheckpoint {
    pub weights: Vec<Vec<Vec<f64>>>,
    pub layer_names: Vec<String>,
    pub layer_sizes: Vec<(usize, usize)>,
    pub neuron_types: Vec<String>,
    pub frozen_layers: Vec<String>,
}

impl SNNCheckpoint {
    pub fn new(
        weights: Vec<Vec<Vec<f64>>>,
        layer_names: Vec<String>,
        layer_sizes: Vec<(usize, usize)>,
        neuron_types: Vec<String>,
        frozen_layers: Vec<String>,
    ) -> Result<Self, String> {
        let state = Self {
            weights,
            layer_names,
            layer_sizes,
            neuron_types,
            frozen_layers,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn n_layers(&self) -> usize {
        self.weights.len()
    }

    pub fn total_params(&self) -> usize {
        self.weights
            .iter()
            .map(|layer| layer.iter().map(Vec::len).sum::<usize>())
            .sum()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.weights.len() != self.layer_names.len() {
            return Err("weights length must match layer_names".to_string());
        }
        if self.layer_sizes.len() != self.layer_names.len() {
            return Err("layer_sizes length must match layer_names".to_string());
        }
        let unique_names: HashSet<&String> = self.layer_names.iter().collect();
        if unique_names.len() != self.layer_names.len() {
            return Err("layer_names must be unique".to_string());
        }
        if !self.neuron_types.is_empty() && self.neuron_types.len() != self.layer_names.len() {
            return Err("neuron_types length must match layer_names".to_string());
        }
        for frozen in &self.frozen_layers {
            if !unique_names.contains(frozen) {
                return Err("frozen_layers must reference known layers".to_string());
            }
        }
        for (idx, layer) in self.weights.iter().enumerate() {
            let (inputs, outputs) = self.layer_sizes[idx];
            if layer.len() != outputs {
                return Err(format!(
                    "layer_{idx} row count must match layer output size"
                ));
            }
            for row in layer {
                if row.len() != inputs {
                    return Err(format!(
                        "layer_{idx} column count must match layer input size"
                    ));
                }
                if row.iter().any(|value| !value.is_finite()) {
                    return Err(format!("layer_{idx} weights must be finite"));
                }
            }
        }
        Ok(())
    }
}

pub fn validate_checkpoint(state: &SNNCheckpoint) -> bool {
    state.validate().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> SNNCheckpoint {
        SNNCheckpoint::new(
            vec![vec![vec![0.1, 0.2], vec![0.3, 0.4]], vec![vec![0.5, 0.6]]],
            vec!["hidden".to_string(), "output".to_string()],
            vec![(2, 2), (2, 1)],
            vec!["LIF".to_string(), "LIF".to_string()],
            vec!["hidden".to_string()],
        )
        .unwrap()
    }

    #[test]
    fn test_checkpoint_counts_parameters() {
        let state = fixture();
        assert_eq!(state.n_layers(), 2);
        assert_eq!(state.total_params(), 6);
        assert!(validate_checkpoint(&state));
    }

    #[test]
    fn test_checkpoint_rejects_shape_mismatch() {
        let result = SNNCheckpoint::new(
            vec![vec![vec![0.1, 0.2]]],
            vec!["hidden".to_string()],
            vec![(3, 1)],
            vec![],
            vec![],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_rejects_unknown_frozen_layer() {
        let result = SNNCheckpoint::new(
            vec![vec![vec![0.1]]],
            vec!["hidden".to_string()],
            vec![(1, 1)],
            vec![],
            vec!["missing".to_string()],
        );
        assert!(result.is_err());
    }
}
