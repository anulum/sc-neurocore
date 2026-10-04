// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fallible packed HDC output allocation

//! Preserve packed HDC algebra while allocating outputs before mutation.

use std::collections::TryReserveError;

use super::BitStreamTensor;

/// Reserve the complete output capacity without infallible growth.
fn reserve<T>(length: usize) -> Result<Vec<T>, TryReserveError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length)?;
    Ok(values)
}

impl BitStreamTensor {
    /// Copy packed storage, returning a reservation error without mutation.
    pub fn try_clone(&self) -> Result<Self, TryReserveError> {
        let mut data = reserve(self.data.len())?;
        data.extend_from_slice(&self.data);
        Ok(Self {
            data,
            length: self.length,
        })
    }

    /// XOR equal-length tensors, returning an error if output storage is unavailable.
    pub fn try_xor(&self, other: &Self) -> Result<Self, TryReserveError> {
        assert_eq!(
            self.length, other.length,
            "Bitstream lengths must match for XOR."
        );
        let mut data = reserve(self.data.len().min(other.data.len()))?;
        data.extend(self.data.iter().zip(&other.data).map(|(&a, &b)| a ^ b));
        Ok(Self {
            data,
            length: self.length,
        })
    }

    /// Rotate logical bits, leaving the tensor unchanged if either reservation fails.
    pub fn try_rotate_right(&mut self, shift: usize) -> Result<(), TryReserveError> {
        if self.length == 0 || shift.is_multiple_of(self.length) {
            return Ok(());
        }
        let mut bits = reserve(self.length)?;
        bits.extend(
            (0..self.length).map(|index| ((self.data[index / 64] >> (index % 64)) & 1) as u8),
        );
        bits.rotate_right(shift % self.length);
        let mut data = reserve(self.length.div_ceil(64))?;
        data.resize(self.length.div_ceil(64), 0_u64);
        for (index, bit) in bits.iter().copied().enumerate() {
            if bit != 0 {
                data[index / 64] |= 1_u64 << (index % 64);
            }
        }
        self.data = data;
        Ok(())
    }

    /// Bundle packed tensors by strict majority, reserving the complete output first.
    pub fn try_bundle(vectors: &[&Self]) -> Result<Self, TryReserveError> {
        assert!(!vectors.is_empty(), "Cannot bundle zero vectors.");
        if vectors.len() == 1 {
            return vectors[0].try_clone();
        }
        let length = vectors[0].length;
        let words = vectors[0].data.len();
        let mut data = reserve(words)?;
        data.resize(words, 0_u64);
        if vectors.len() == 3 {
            for (index, item) in data.iter_mut().enumerate() {
                let a = vectors[0].data[index];
                let b = vectors[1].data[index];
                let c = vectors[2].data[index];
                *item = (a & b) | (b & c) | (a & c);
            }
        } else {
            let threshold = vectors.len() / 2;
            for (index, item) in data.iter_mut().enumerate() {
                for bit in 0..64 {
                    let mut count = 0;
                    for vector in vectors {
                        if (vector.data[index] >> bit) & 1 == 1 {
                            count += 1;
                        }
                    }
                    if count > threshold {
                        *item |= 1_u64 << bit;
                    }
                }
            }
        }
        Ok(Self { data, length })
    }
}
