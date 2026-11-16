// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Fixed-point LIF step

use super::masking::{mask, mask_wide};
use super::FixedPointLif;

impl FixedPointLif {
    /// Advance one fixed-point step and return the spike and masked voltage.
    ///
    /// While refractory, the counter counts down, `v` is held at `v_rest` and the step
    /// returns 0. Otherwise `v` gains `((v_rest - v) * leak_k) >> fraction`,
    /// `(i_t * gain_k) >> fraction` and `noise_in`; the leak and input terms and the sum
    /// are each masked to `data_width` bits (`v_rest - v` to twice that). Reaching
    /// `v_threshold` sets `v` to `v_reset`, starts the refractory count and returns 1.
    /// The returned voltage is the new `v` masked to `data_width` bits.
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
}
