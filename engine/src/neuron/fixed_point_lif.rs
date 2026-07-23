// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point leaky-integrate-and-fire neuron

/// Mask and sign-interpret an integer to `width` bits (branchless).
#[inline]
pub fn mask(value: i32, width: u32) -> i16 {
    assert!(
        width > 0 && width <= 32,
        "mask width must be 1..=32, got {width}"
    );
    let mask = (1_i64 << width) - 1;
    let value = (value as i64) & mask;
    let shift = 64 - width;
    ((value << shift) >> shift) as i16
}

/// Fixed-point leaky-integrate-and-fire neuron state and parameters.
#[derive(Clone, Debug)]
pub struct FixedPointLif {
    pub v: i16,
    pub refractory_counter: i32,
    pub data_width: u32,
    pub fraction: u32,
    pub v_rest: i16,
    pub v_reset: i16,
    pub v_threshold: i16,
    pub refractory_period: i32,
}

impl FixedPointLif {
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

    #[allow(non_snake_case)]
    pub fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let width = self.data_width;
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            return (0, mask(self.v_rest as i32, width));
        }

        let diff = mask((self.v_rest as i32) - (self.v as i32), 2 * width) as i32;
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
            (1, mask(self.v_reset as i32, width))
        } else {
            self.v = v_next;
            (0, mask(v_next as i32, width))
        }
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{mask, FixedPointLif};

    #[test]
    fn branchless_mask_matches_signed_reference() {
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
                let bit_mask = (1_i64 << width) - 1;
                let mut expected = (value as i64) & bit_mask;
                if expected >= (1_i64 << (width - 1)) {
                    expected -= 1_i64 << width;
                }
                let expected = if width >= 32 {
                    expected as i32 as i16
                } else {
                    expected as i16
                };
                assert_eq!(result, expected, "value={value}, width={width}");
            }
        }
    }

    #[test]
    fn refractory_period_enforces_two_silent_steps() {
        let mut neuron = FixedPointLif::new(16, 8, 0, 0, 256, 2);
        let spikes: Vec<_> = (0..30).map(|_| neuron.step(1, 256, 50, 0).0).collect();
        assert!(spikes.iter().sum::<i32>() > 0);
        for (index, &spike) in spikes.iter().enumerate() {
            if spike == 1 && index + 2 < spikes.len() {
                assert_eq!(spikes[index + 1], 0);
                assert_eq!(spikes[index + 2], 0);
            }
        }
    }

    #[test]
    fn zero_refractory_period_allows_repeated_firing() {
        let mut neuron = FixedPointLif::new(16, 8, 0, 0, 256, 0);
        let spikes: i32 = (0..20).map(|_| neuron.step(1, 256, 50, 0).0).sum();
        assert!(spikes > 0);
    }
}
