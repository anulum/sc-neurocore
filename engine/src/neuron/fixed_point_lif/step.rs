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
        let (spike, v_out, _membrane) = self.step_with_membrane(leak_k, gain_k, i_t, noise_in);
        (spike, v_out)
    }

    /// Advance one step as [`Self::step`] does, and also return the membrane before any reset.
    ///
    /// The membrane is the masked sum that was compared with `v_threshold`, so it is at or
    /// above the threshold exactly on a spike step, where the returned voltage is `v_reset`.
    /// On a refractory step nothing is integrated and the membrane is the held `v_rest`.
    pub(crate) fn step_with_membrane(
        &mut self,
        leak_k: i16,
        gain_k: i16,
        i_t: i16,
        noise_in: i16,
    ) -> (i32, i16, i16) {
        let width = self.data_width;
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            return (0, mask(self.v_rest as i32, width), self.v_rest);
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
            (1, mask(self.v_reset as i32, width), v_next)
        } else {
            self.v = v_next;
            (0, mask(v_next as i32, width), v_next)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FixedPointLif;

    /// `step` and `step_with_membrane` agree, and the membrane is the integrated value the
    /// threshold was compared with: above it exactly on a spike step, the voltage otherwise,
    /// and `v_rest` while refractory. A narrow word wraps the sum, and the membrane wraps too.
    #[test]
    fn the_membrane_is_the_value_compared_with_the_threshold() {
        for (width, threshold) in [(16, 256), (8, 64)] {
            let mut plain = FixedPointLif::new(width, 4, 0, -8, threshold, 1);
            let mut traced = plain.clone();
            let (mut spikes, mut refractory_steps) = (0, 0);
            for current in [3_i16, 40, 200, 1000, -50, 700, 5, 300] {
                let (spike, v_out) = plain.step(3, 16, current, 1);
                let refractory = traced.refractory_counter > 0;
                let (spike_m, v_out_m, membrane) = traced.step_with_membrane(3, 16, current, 1);
                assert_eq!((spike, v_out), (spike_m, v_out_m));
                spikes += spike;
                refractory_steps += i32::from(refractory);
                if refractory {
                    assert_eq!((spike, membrane), (0, traced.v_rest));
                } else {
                    assert_eq!(spike == 1, membrane >= threshold);
                    if spike == 0 {
                        assert_eq!(membrane, v_out);
                    }
                }
            }
            assert!(
                spikes > 0 && refractory_steps > 0,
                "width {width}: both branches ran"
            );
        }
    }
}
