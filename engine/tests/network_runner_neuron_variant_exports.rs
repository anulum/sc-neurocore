// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner neuron-variant public contract

use sc_neurocore_engine::network_runner::{create_neuron, NeuronVariant};

#[test]
fn public_variant_dispatch_preserves_step_reset_and_voltage_access() {
    let mut neuron = create_neuron("Izhikevich").expect("Izhikevich must remain supported");
    assert!(matches!(neuron, NeuronVariant::Izhikevich(_)));

    let spike = neuron.step(10.0);
    assert!(matches!(spike, 0 | 1));
    assert!(neuron.soma_voltage().is_finite());

    neuron.reset();
    assert!(neuron.soma_voltage().is_finite());
}

#[test]
fn resonate_and_fire_variant_reports_the_source_voltage_coordinate() {
    let mut neuron =
        create_neuron("ResonateAndFireNeuron").expect("resonate-and-fire must remain supported");
    assert_eq!(neuron.step(5.0), 0);
    let reported = neuron.soma_voltage();

    match neuron {
        NeuronVariant::ResonateAndFire(inner) => {
            assert_eq!(reported, inner.y);
            assert_ne!(inner.x, inner.y);
        }
        _ => panic!("factory returned the wrong neuron variant"),
    }
}
