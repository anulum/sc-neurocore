// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DPI neuron parameter and state contract

use super::DPINeuron;

impl DPINeuron {
    pub(super) fn valid(&self) -> bool {
        self.i_mem.is_finite()
            && self.i_mem > 0.0
            && self.i_ahp.is_finite()
            && self.i_ahp >= 0.0
            && self.refractory_time.is_finite()
            && self.refractory_time >= 0.0
            && self.i_threshold.is_finite()
            && self.i_threshold > 0.0
            && self.i_reset.is_finite()
            && self.i_reset > 0.0
            && self.i_reset < self.i_threshold
            && self.i_rest.is_finite()
            && self.i_rest >= 0.0
            && self.i_tau.is_finite()
            && self.i_tau > 0.0
            && self.i_g.is_finite()
            && self.i_g > 0.0
            && self.i_tau_ahp.is_finite()
            && self.i_tau_ahp > 0.0
            && self.i_ga.is_finite()
            && self.i_ga > 0.0
            && self.i_spike.is_finite()
            && self.i_spike > 0.0
            && self.i_0.is_finite()
            && self.i_0 > 0.0
            && self.kappa.is_finite()
            && self.kappa > 0.0
            && self.alpha.is_finite()
            && self.alpha > 0.0
            && self.tau.is_finite()
            && self.tau > 0.0
            && self.tau_ahp.is_finite()
            && self.tau_ahp > 0.0
            && self.refractory_period.is_finite()
            && self.refractory_period > 0.0
            && self.dt.is_finite()
            && self.dt > 0.0
            && self.refractory_period >= self.dt
    }
}
