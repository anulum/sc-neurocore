// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Owned dense IF parameters and per-layer membrane preloads

//! Checked dense coefficients, activation modes and domain errors.

/// A shape, domain or finite-arithmetic refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayError {
    /// Invalid coefficients, input frames or coupled state dimensions.
    InvalidInput,
    /// Finite arithmetic produced a nonfinite state or current.
    Overflow,
    /// Addressable replay buffers exceed the declared byte budget.
    ResourceLimit,
}

/// A final IF spike count or signed linear integrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Inclusive-threshold IF events with subtractive reset.
    Spikes,
    /// Signed accumulated final-layer current without IF events.
    Linear,
}

/// Owned row-major output-by-input coefficients for one layer.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub(super) inputs: usize,
    pub(super) outputs: usize,
    pub(super) weights: Vec<f64>,
    pub(super) bias: Option<Vec<f64>>,
    pub(super) threshold: f64,
    pub(super) initial_fraction: Option<f64>,
}

impl DenseLayer {
    /// Validate dimensions, finite coefficients and a positive finite threshold.
    pub fn new(
        inputs: usize,
        outputs: usize,
        weights: Vec<f64>,
        bias: Option<Vec<f64>>,
        threshold: f64,
    ) -> Result<Self, ReplayError> {
        if inputs == 0
            || outputs == 0
            || inputs.checked_mul(outputs) != Some(weights.len())
            || weights.iter().any(|v| !v.is_finite())
            || bias
                .as_ref()
                .is_some_and(|v| v.len() != outputs || v.iter().any(|x| !x.is_finite()))
            || !threshold.is_finite()
            || threshold <= 0.0
        {
            return Err(ReplayError::InvalidInput);
        }
        Ok(Self {
            inputs,
            outputs,
            weights,
            bias,
            threshold,
            initial_fraction: None,
        })
    }
    /// Set this layer's finite preload independently of the global fallback.
    pub fn with_initial_fraction(mut self, fraction: f64) -> Result<Self, ReplayError> {
        if !fraction.is_finite() {
            return Err(ReplayError::InvalidInput);
        }
        self.initial_fraction = Some(fraction);
        Ok(self)
    }
}
