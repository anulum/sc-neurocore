// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Lapicque neuron configuration and state contract

use super::LapicqueNeuron;

impl LapicqueNeuron {
    /// Return whether all configuration and dynamic invariants hold.
    ///
    /// Both profiles need a finite `v` and a finite, positive `v_threshold` and `dt`. The
    /// source profile also needs a finite, positive `capacitance`, `series_resistance`
    /// and `polarization_resistance`, and `v` below `v_threshold` until `excited` is set.
    /// The SC profile needs `excited` clear, a finite `v_rest` and `v_reset` below
    /// `v_threshold`, `v` below `v_threshold`, and a finite, positive `tau` and
    /// `resistance`.
    pub fn valid(&self) -> bool {
        if !self.v.is_finite()
            || !self.v_threshold.is_finite()
            || self.v_threshold <= 0.0
            || !self.dt.is_finite()
            || self.dt <= 0.0
        {
            return false;
        }
        if self.source_profile {
            return (self.excited || self.v < self.v_threshold)
                && self.capacitance.is_finite()
                && self.capacitance > 0.0
                && self.series_resistance.is_finite()
                && self.series_resistance > 0.0
                && self.polarization_resistance.is_finite()
                && self.polarization_resistance > 0.0;
        }
        !self.excited
            && self.v_rest.is_finite()
            && self.v_reset.is_finite()
            && self.v_threshold > self.v_rest
            && self.v_threshold > self.v_reset
            && self.v < self.v_threshold
            && self.tau.is_finite()
            && self.tau > 0.0
            && self.resistance.is_finite()
            && self.resistance > 0.0
    }
}
