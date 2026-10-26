// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Dense-layer allocation errors

//! Checked dimensions and fallible storage for stochastic dense layers.

use std::collections::TryReserveError;
use std::fmt;

/// Distinguish authored input refusals from dense storage exhaustion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DenseLayerError {
    /// A dimension or input shape violates the layer contract.
    Invalid(String),
    /// The requested capacity or allocation cannot be represented or reserved.
    Allocation,
}

impl fmt::Display for DenseLayerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => formatter.write_str(message),
            Self::Allocation => formatter.write_str("dense storage cannot be allocated"),
        }
    }
}

impl std::error::Error for DenseLayerError {}

impl From<TryReserveError> for DenseLayerError {
    fn from(_: TryReserveError) -> Self {
        Self::Allocation
    }
}

/// Multiply storage dimensions without wrapping before a reservation.
pub(super) fn product(left: usize, right: usize) -> Result<usize, DenseLayerError> {
    left.checked_mul(right).ok_or(DenseLayerError::Allocation)
}

/// Reserve all numeric elements before initializing them.
pub(super) fn zeroed<T: Default + Clone>(length: usize) -> Result<Vec<T>, DenseLayerError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length)?;
    values.resize(length, T::default());
    Ok(values)
}

/// Independently copy every probability row after fallible reservations.
pub(super) fn copy_weights(weights: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, DenseLayerError> {
    let mut copied = Vec::new();
    copied.try_reserve_exact(weights.len())?;
    for row in weights {
        let mut owned = Vec::new();
        owned.try_reserve_exact(row.len())?;
        owned.extend_from_slice(row);
        copied.push(owned);
    }
    Ok(copied)
}
