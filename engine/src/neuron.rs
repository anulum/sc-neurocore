// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Neuron Models

//! # Neuron Models
//!
//! Fixed-point LIF and Izhikevich neuron models for the v3 engine.

mod adex;
mod bitstream_averager;
mod dendritic_neuron;
mod exp_if;
mod homeostatic_lif;
mod izhikevich;
mod lapicque;

pub use adex::AdExNeuron;
pub use bitstream_averager::BitstreamAverager;
pub use dendritic_neuron::DendriticNeuron;
pub use exp_if::ExpIfNeuron;
pub use homeostatic_lif::HomeostaticLif;
pub use izhikevich::Izhikevich;
pub use lapicque::LapicqueNeuron;

/// Mask and sign-interpret an integer to `width` bits (branchless).
///
/// `width` must be in 1..=32. Values outside this range trigger a
/// debug assertion failure (release builds silently produce garbage).
#[inline]
pub fn mask(value: i32, width: u32) -> i16 {
    assert!(
        width > 0 && width <= 32,
        "mask width must be 1..=32, got {width}"
    );
    let m = (1_i64 << width) - 1;
    let v = (value as i64) & m;
    let shift = 64 - width;
    ((v << shift) >> shift) as i16
}

/// Fixed-point leaky-integrate-and-fire neuron state and parameters.
#[derive(Clone, Debug)]
pub struct FixedPointLif {
    /// Membrane potential.
    pub v: i16,
    /// Refractory counter in simulation steps.
    pub refractory_counter: i32,
    /// Arithmetic data width.
    pub data_width: u32,
    /// Fraction bits for fixed-point scaling.
    pub fraction: u32,
    /// Resting potential.
    pub v_rest: i16,
    /// Reset potential after spike.
    pub v_reset: i16,
    /// Spike threshold.
    pub v_threshold: i16,
    /// Refractory period length in steps.
    pub refractory_period: i32,
}

impl FixedPointLif {
    /// Construct a fixed-point LIF neuron.
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
    ) -> Self {
        Self {
            v: v_rest,
            refractory_counter: 0,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        }
    }

    /// Advance one simulation step.
    ///
    /// Returns `(spike, membrane_voltage)`.
    #[allow(non_snake_case)]
    pub fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let w = self.data_width;

        // Refractory: check previous step's counter before any fire logic.
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            return (0, mask(self.v_rest as i32, w));
        }

        let diff = mask((self.v_rest as i32) - (self.v as i32), 2 * w) as i32;
        let dv_leak = mask((diff * (leak_k as i32)) >> self.fraction, self.data_width);
        let dv_in = mask(
            ((i_t as i32) * (gain_k as i32)) >> self.fraction,
            self.data_width,
        );

        let v_next = mask(
            (self.v as i32) + (dv_leak as i32) + (dv_in as i32) + (noise_in as i32),
            self.data_width,
        );

        if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_counter = self.refractory_period;
            (1, mask(self.v_reset as i32, w))
        } else {
            self.v = v_next;
            (0, mask(v_next as i32, w))
        }
    }

    /// Reset internal state to resting potential.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{mask, FixedPointLif};

    #[test]
    fn mask_branchless_matches_original() {
        for &width in &[16_u32, 32] {
            for value in [
                -32768_i32,
                -1,
                0,
                1,
                32767,
                65535,
                -65536,
                i16::MAX as i32,
                i16::MIN as i32,
            ] {
                let result = mask(value, width);

                let m = (1_i64 << width) - 1;
                let mut v = (value as i64) & m;
                if v >= (1_i64 << (width - 1)) {
                    v -= 1_i64 << width;
                }
                let expected = if width >= 32 {
                    v as i32 as i16
                } else {
                    v as i16
                };

                assert_eq!(
                    result, expected,
                    "mask({value}, {width}): got {result}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn lif_fires_with_refractory_period() {
        // Q8.8: threshold=1.0 → 256, matching Python default
        let mut n = FixedPointLif::new(16, 8, 0, 0, 256, 2);
        let mut spikes = Vec::new();
        for _ in 0..30 {
            let (s, _) = n.step(1, 256, 50, 0);
            spikes.push(s);
        }
        let total: i32 = spikes.iter().sum();
        assert!(total > 0, "neuron must fire with refractory_period=2");
        // Refractory gap: after a spike, next 2 steps must be silent.
        for (i, &s) in spikes.iter().enumerate() {
            if s == 1 && i + 2 < spikes.len() {
                assert_eq!(spikes[i + 1], 0, "step {} should be refractory", i + 1);
                assert_eq!(spikes[i + 2], 0, "step {} should be refractory", i + 2);
            }
        }
    }

    #[test]
    fn lif_fires_without_refractory() {
        let mut n = FixedPointLif::new(16, 8, 0, 0, 256, 0);
        let mut total = 0;
        for _ in 0..20 {
            let (s, _) = n.step(1, 256, 50, 0);
            total += s;
        }
        assert!(total > 0, "neuron must fire with refractory_period=0");
    }
}
