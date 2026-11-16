// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Akida Neuron Emulator

#![deny(missing_docs)]

/// Akida — BrainChip event-domain rank-order neuron.
#[derive(Clone, Debug)]
pub struct AkidaNeuron {
    /// Integer membrane accumulator.
    pub v: i32,
    /// Membrane threshold for an output spike (integer accumulator units).
    pub threshold: i32,
    /// Rank-order decay factor: each input event since the last reset is weighted by one more factor of modulation than the one before it (dimensionless).
    pub modulation: f64,
    /// Input events accepted since the last reset; the next event is weighted by
    /// `modulation^rank`.
    pub rank: i32,
    /// Whether the neuron has fired since the last reset; it fires at most once per
    /// presentation.
    pub spiked: bool,
}

impl AkidaNeuron {
    /// Create a neuron with the given threshold and `modulation` = 0.75.
    pub fn new(threshold: i32) -> Self {
        Self {
            v: 0,
            threshold,
            modulation: 0.75,
            rank: 0,
            spiked: false,
        }
    }
    /// Accept one input event and return 1 when the accumulator reaches `threshold`, else 0.
    ///
    /// The event adds `weight * modulation^rank`, truncated towards zero to an integer (the
    /// cast saturates; NaN adds 0), and the rank grows by 1. After the first spike, every
    /// later event is ignored and returns 0 until `reset`.
    pub fn step(&mut self, weight: f64) -> i32 {
        if self.spiked {
            return 0;
        }
        self.v += (weight * self.modulation.powi(self.rank)) as i32;
        self.rank += 1;
        if self.v >= self.threshold {
            self.spiked = true;
            1
        } else {
            0
        }
    }
    /// Clear the accumulator, the rank and the spiked flag; the parameters are kept.
    pub fn reset(&mut self) {
        self.v = 0;
        self.rank = 0;
        self.spiked = false;
    }
}
impl Default for AkidaNeuron {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
#[path = "akida_tests.rs"]
mod tests;
