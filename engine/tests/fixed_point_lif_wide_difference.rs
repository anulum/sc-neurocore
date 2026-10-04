// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public fixed-point LIF wide subtraction regressions

use sc_neurocore_engine::neuron::FixedPointLif;

#[test]
fn positive_wide_difference_retains_bits_before_fractional_scaling() {
    let mut neuron = FixedPointLif::new(16, 8, i16::MAX, 0, i16::MAX, 0);
    assert_eq!(neuron.step(1, 256, i16::MIN, 0), (0, -1));
    assert_eq!(neuron.step(1, 256, 0, 0), (0, 127));
    assert_eq!(neuron.v, 127);
    assert_eq!(neuron.refractory_counter, 0);
    neuron.reset();
    assert_eq!(neuron.v, i16::MAX);
    assert_eq!(neuron.step(1, 256, i16::MIN, 0), (0, -1));
}

#[test]
fn negative_wide_difference_retains_bits_before_fractional_scaling() {
    let mut neuron = FixedPointLif::new(16, 8, i16::MIN, 0, i16::MAX, 0);
    assert_eq!(neuron.step(1, 256, -2, 0), (0, 32766));
    assert_eq!(neuron.step(1, 256, 0, 0), (0, 32510));
    assert_eq!(neuron.v, 32510);
    assert_eq!(neuron.refractory_counter, 0);
}
