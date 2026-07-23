// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point Q-format contracts

use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QFormat {
    pub integer_bits: u8,
    pub fraction_bits: u8,
}

impl QFormat {
    pub const fn q8_8() -> Self {
        Self {
            integer_bits: 8,
            fraction_bits: 8,
        }
    }

    pub const fn q16_16() -> Self {
        Self {
            integer_bits: 16,
            fraction_bits: 16,
        }
    }

    pub fn new(integer_bits: u8, fraction_bits: u8) -> Result<Self, QFormatError> {
        if integer_bits == 0 {
            return Err(QFormatError::MissingSignBit);
        }
        let total_bits = u16::from(integer_bits) + u16::from(fraction_bits);
        if total_bits == 0 || total_bits > 63 {
            return Err(QFormatError::TotalBitsTooWide(total_bits));
        }
        Ok(Self {
            integer_bits,
            fraction_bits,
        })
    }

    pub fn total_bits(self) -> u8 {
        self.integer_bits + self.fraction_bits
    }

    pub fn scale(self) -> i128 {
        1_i128 << self.fraction_bits
    }

    pub fn min_value(self) -> f64 {
        -((1_i128 << (self.total_bits() - 1)) as f64) / self.scale() as f64
    }

    pub fn max_value(self) -> f64 {
        ((1_i128 << (self.total_bits() - 1)) - 1) as f64 / self.scale() as f64
    }

    pub fn label(self) -> String {
        format!("Q{}.{}", self.integer_bits, self.fraction_bits)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QFormatError {
    MissingSignBit,
    TotalBitsTooWide(u16),
    AccumulatorNarrower,
    AccumulatorFractionLoss,
    AccumulatorRangeLoss,
}

impl fmt::Display for QFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSignBit => write!(f, "integer_bits must include the sign bit"),
            Self::TotalBitsTooWide(bits) => {
                write!(f, "Q-format total bits exceed i64 range: {bits}")
            }
            Self::AccumulatorNarrower => write!(
                f,
                "accumulator format must not be narrower than weight format"
            ),
            Self::AccumulatorFractionLoss => {
                write!(
                    f,
                    "accumulator format must preserve weight fractional precision"
                )
            }
            Self::AccumulatorRangeLoss => {
                write!(f, "accumulator format must cover the full weight range")
            }
        }
    }
}

impl Error for QFormatError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QFormatMixed {
    pub weight_fmt: QFormat,
    pub accum_fmt: QFormat,
    pub scale_per_tensor: bool,
}

impl QFormatMixed {
    pub fn q8_8_q16_16() -> Self {
        Self {
            weight_fmt: QFormat::q8_8(),
            accum_fmt: QFormat::q16_16(),
            scale_per_tensor: true,
        }
    }

    pub fn new(
        weight_fmt: QFormat,
        accum_fmt: QFormat,
        scale_per_tensor: bool,
    ) -> Result<Self, QFormatError> {
        if accum_fmt.total_bits() < weight_fmt.total_bits() {
            return Err(QFormatError::AccumulatorNarrower);
        }
        if accum_fmt.fraction_bits < weight_fmt.fraction_bits {
            return Err(QFormatError::AccumulatorFractionLoss);
        }
        if accum_fmt.min_value() > weight_fmt.min_value()
            || accum_fmt.max_value() < weight_fmt.max_value()
        {
            return Err(QFormatError::AccumulatorRangeLoss);
        }
        Ok(Self {
            weight_fmt,
            accum_fmt,
            scale_per_tensor,
        })
    }

    pub fn accumulator_guard_bits(self) -> u8 {
        self.accum_fmt.total_bits() - self.weight_fmt.total_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qformat_mixed_default_matches_python_contract() {
        let fmt = QFormatMixed::q8_8_q16_16();

        assert_eq!(fmt.weight_fmt.label(), "Q8.8");
        assert_eq!(fmt.accum_fmt.label(), "Q16.16");
        assert_eq!(fmt.accumulator_guard_bits(), 16);
    }

    #[test]
    fn rejects_accumulator_precision_loss() {
        let result = QFormatMixed::new(
            QFormat::new(8, 12).unwrap(),
            QFormat::new(16, 8).unwrap(),
            true,
        );

        assert_eq!(result.unwrap_err(), QFormatError::AccumulatorFractionLoss);
    }
}
