// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Rust safety mirror for adaptive precision rows

#[derive(Debug, Clone, PartialEq)]
pub struct LayerPrecision {
    pub layer_index: usize,
    pub name: String,
    pub bitstream_length: usize,
    pub error_bound: f64,
    pub sensitivity: f64,
}

impl LayerPrecision {
    pub fn new(
        layer_index: usize,
        name: String,
        bitstream_length: usize,
        error_bound: f64,
        sensitivity: f64,
    ) -> Result<Self, String> {
        let row = Self {
            layer_index,
            name,
            bitstream_length,
            error_bound,
            sensitivity,
        };
        row.validate()?;
        Ok(row)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("name must be a non-empty string".to_string());
        }
        if self.bitstream_length == 0 {
            return Err("bitstream_length must be positive".to_string());
        }
        if !self.bitstream_length.is_power_of_two() {
            return Err("bitstream_length must be a power of two".to_string());
        }
        validate_non_negative(self.error_bound, "error_bound")?;
        validate_non_negative(self.sensitivity, "sensitivity")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SynapsePrecision {
    pub layer_index: usize,
    pub layer_name: String,
    pub output_index: usize,
    pub input_index: usize,
    pub bit_width: usize,
    pub bitstream_length: usize,
    pub sensitivity: f64,
    pub quantization_error_bound: f64,
    pub stochastic_error_bound: f64,
    pub total_error_bound: f64,
}

impl SynapsePrecision {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer_index: usize,
        layer_name: String,
        output_index: usize,
        input_index: usize,
        bit_width: usize,
        bitstream_length: usize,
        sensitivity: f64,
        quantization_error_bound: f64,
        stochastic_error_bound: f64,
        total_error_bound: f64,
    ) -> Result<Self, String> {
        let row = Self {
            layer_index,
            layer_name,
            output_index,
            input_index,
            bit_width,
            bitstream_length,
            sensitivity,
            quantization_error_bound,
            stochastic_error_bound,
            total_error_bound,
        };
        row.validate()?;
        Ok(row)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.layer_name.is_empty() {
            return Err("layer_name must be a non-empty string".to_string());
        }
        if self.bit_width == 0 {
            return Err("bit_width must be positive".to_string());
        }
        if self.bitstream_length == 0 {
            return Err("bitstream_length must be positive".to_string());
        }
        validate_non_negative(self.sensitivity, "sensitivity")?;
        validate_non_negative(self.quantization_error_bound, "quantization_error_bound")?;
        validate_non_negative(self.stochastic_error_bound, "stochastic_error_bound")?;
        validate_non_negative(self.total_error_bound, "total_error_bound")?;
        let component_sum = self.quantization_error_bound + self.stochastic_error_bound;
        if self.total_error_bound + 1e-15 < component_sum {
            return Err("total_error_bound must cover component bounds".to_string());
        }
        Ok(())
    }
}

pub fn validate_adaptive_precision(row: &LayerPrecision) -> bool {
    row.validate().is_ok()
}

pub fn validate_synapse_precision(row: &SynapsePrecision) -> bool {
    row.validate().is_ok()
}

fn validate_non_negative(value: f64, name: &str) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and non-negative"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_precision_contract() {
        let row = LayerPrecision::new(0, "fc1".to_string(), 256, 0.03125, 0.5).unwrap();
        assert!(validate_adaptive_precision(&row));
        assert!(LayerPrecision::new(0, "fc1".to_string(), 300, 0.03125, 0.5).is_err());
        assert!(LayerPrecision::new(0, "".to_string(), 256, 0.03125, 0.5).is_err());
    }

    #[test]
    fn test_synapse_precision_contract() {
        let row = SynapsePrecision::new(
            0,
            "fc1".to_string(),
            1,
            2,
            8,
            128,
            0.5,
            0.01,
            0.02,
            0.03,
        )
        .unwrap();
        assert!(validate_synapse_precision(&row));
        assert!(SynapsePrecision::new(
            0,
            "fc1".to_string(),
            1,
            2,
            8,
            128,
            0.5,
            0.02,
            0.02,
            0.03,
        )
        .is_err());
    }
}
