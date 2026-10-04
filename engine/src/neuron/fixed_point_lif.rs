// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point leaky-integrate-and-fire neuron

/// Mask and sign-interpret an integer to `width` bits, then narrow to `i16`.
///
/// Widths above 16 retain the low 16 bits in this public narrow-value helper.
#[inline]
pub fn mask(value: i32, width: u32) -> i16 {
    mask_wide(value, width) as i16
}

/// Interpret a wide intermediate without narrowing before fractional scaling.
#[inline]
fn mask_wide(value: i32, width: u32) -> i32 {
    assert!(
        width > 0 && width <= 32,
        "mask width must be 1..=32, got {width}"
    );
    let shift = 32 - width;
    value.wrapping_shl(shift) >> shift
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
    /// Construct a neuron with valid native fixed-point parameters.
    ///
    /// # Panics
    ///
    /// Panics for invalid parameters. Use [`Self::try_new`] for fallible input.
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
    ) -> Self {
        Self::try_new(
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        )
        .expect("invalid fixed-point LIF configuration")
    }

    /// Construct a neuron without accepting configurations that cannot run.
    ///
    /// # Errors
    ///
    /// Rejects widths outside the native `i16` range, fractions at or above the
    /// state width, and negative refractory periods before constructing state.
    pub fn try_new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
    ) -> Result<Self, &'static str> {
        if !(1..=16).contains(&data_width) {
            return Err("data_width must be in [1, 16]");
        }
        if fraction >= data_width {
            return Err("fraction must be less than data_width");
        }
        if refractory_period < 0 {
            return Err("refractory_period must be nonnegative");
        }
        Ok(Self {
            v: v_rest,
            refractory_counter: 0,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        })
    }

    /// Advance one fixed-point step and return the spike and masked voltage.
    #[allow(non_snake_case)]
    pub fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let width = self.data_width;
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            return (0, mask(self.v_rest as i32, width));
        }

        let diff = mask_wide((self.v_rest as i32) - (self.v as i32), 2 * width);
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

    /// Restore resting voltage and clear the refractory counter.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_counter = 0;
    }
}

#[cfg(test)]
mod tests;
