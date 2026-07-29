// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for transfer fine tuning

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct TransferCheckpoint {
    pub layer_names: Vec<String>,
    pub frozen_layers: Vec<String>,
}

impl TransferCheckpoint {
    pub fn new(layer_names: Vec<String>, frozen_layers: Vec<String>) -> Result<Self, String> {
        let mut state = Self {
            layer_names,
            frozen_layers,
        };
        state.validate()?;
        state.frozen_layers.sort();
        state.frozen_layers.dedup();
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), String> {
        let unique_names: HashSet<&String> = self.layer_names.iter().collect();
        if unique_names.len() != self.layer_names.len() {
            return Err("layer_names must be unique".to_string());
        }
        for frozen in &self.frozen_layers {
            if !unique_names.contains(frozen) {
                return Err("frozen_layers must reference known layers".to_string());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FreezeUntil {
    None,
    Index(usize),
    Layer(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransferConfig {
    pub freeze_until: FreezeUntil,
    pub lr_backbone: f64,
    pub lr_head: f64,
}

impl TransferConfig {
    pub fn new(freeze_until: FreezeUntil, lr_backbone: f64, lr_head: f64) -> Result<Self, String> {
        if !lr_backbone.is_finite() || !lr_head.is_finite() || lr_backbone < 0.0 || lr_head < 0.0 {
            return Err("learning rates must be finite and non-negative".to_string());
        }
        Ok(Self {
            freeze_until,
            lr_backbone,
            lr_head,
        })
    }
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            freeze_until: FreezeUntil::None,
            lr_backbone: 0.0,
            lr_head: 0.01,
        }
    }
}

pub fn freeze_layers(
    checkpoint: &mut TransferCheckpoint,
    layer_names: &[&str],
    until_index: Option<usize>,
) -> Result<(), String> {
    validate_layer_names(checkpoint, layer_names)?;
    let mut frozen: HashSet<String> = checkpoint.frozen_layers.iter().cloned().collect();
    for name in layer_names {
        frozen.insert((*name).to_string());
    }
    if let Some(index) = until_index {
        if index >= checkpoint.layer_names.len() {
            return Err("until_index must reference an existing layer".to_string());
        }
        for name in checkpoint.layer_names.iter().take(index + 1) {
            frozen.insert(name.clone());
        }
    }
    checkpoint.frozen_layers = frozen.into_iter().collect();
    checkpoint.frozen_layers.sort();
    Ok(())
}

pub fn unfreeze_layers(
    checkpoint: &mut TransferCheckpoint,
    layer_names: &[&str],
    all_layers: bool,
) -> Result<(), String> {
    if all_layers {
        checkpoint.frozen_layers.clear();
        return Ok(());
    }
    validate_layer_names(checkpoint, layer_names)?;
    let removals: HashSet<&str> = layer_names.iter().copied().collect();
    checkpoint
        .frozen_layers
        .retain(|name| !removals.contains(name.as_str()));
    Ok(())
}

pub fn apply_transfer_config(
    checkpoint: &mut TransferCheckpoint,
    config: &TransferConfig,
) -> Result<Vec<f64>, String> {
    match &config.freeze_until {
        FreezeUntil::None => {}
        FreezeUntil::Index(index) => freeze_layers(checkpoint, &[], Some(*index))?,
        FreezeUntil::Layer(name) => {
            let index = checkpoint
                .layer_names
                .iter()
                .position(|layer| layer == name)
                .ok_or_else(|| "freeze_until layer is not present in checkpoint".to_string())?;
            freeze_layers(checkpoint, &[], Some(index))?;
        }
    }
    Ok(checkpoint
        .layer_names
        .iter()
        .map(|name| {
            if checkpoint.frozen_layers.contains(name) {
                config.lr_backbone
            } else {
                config.lr_head
            }
        })
        .collect())
}

pub fn validate_fine_tune(config: &TransferConfig) -> bool {
    config.lr_backbone.is_finite()
        && config.lr_head.is_finite()
        && config.lr_backbone >= 0.0
        && config.lr_head >= 0.0
}

fn validate_layer_names(
    checkpoint: &TransferCheckpoint,
    layer_names: &[&str],
) -> Result<(), String> {
    let known: HashSet<&String> = checkpoint.layer_names.iter().collect();
    for name in layer_names {
        if !known.contains(&name.to_string()) {
            return Err("Unknown layer names".to_string());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> TransferCheckpoint {
        TransferCheckpoint::new(vec!["hidden".to_string(), "output".to_string()], vec![]).unwrap()
    }

    #[test]
    fn test_apply_transfer_config_freezes_prefix() {
        let mut checkpoint = fixture();
        let config = TransferConfig::new(FreezeUntil::Index(0), 0.0, 0.01).unwrap();
        let lrs = apply_transfer_config(&mut checkpoint, &config).unwrap();
        assert_eq!(checkpoint.frozen_layers, vec!["hidden"]);
        assert_eq!(lrs, vec![0.0, 0.01]);
    }

    #[test]
    fn test_freeze_rejects_unknown_layer() {
        let mut checkpoint = fixture();
        assert!(freeze_layers(&mut checkpoint, &["missing"], None).is_err());
    }

    #[test]
    fn test_config_rejects_bad_learning_rates() {
        assert!(TransferConfig::new(FreezeUntil::None, f64::NAN, 0.01).is_err());
        assert!(TransferConfig::new(FreezeUntil::None, 0.0, -0.01).is_err());
    }
}
