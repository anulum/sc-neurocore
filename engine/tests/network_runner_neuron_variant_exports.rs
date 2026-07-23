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

#[test]
fn public_variant_supports_long_single_neuron_simulation() {
    let mut neuron = create_neuron("AdEx").expect("AdEx must remain supported");
    let mut voltages = Vec::with_capacity(1_000);
    let mut spike_count = 0;

    for _ in 0..1_000 {
        spike_count += usize::from(neuron.step(500.0) != 0);
        voltages.push(neuron.soma_voltage());
    }

    assert_eq!(voltages.len(), 1_000);
    assert!(voltages.iter().all(|voltage| voltage.is_finite()));
    assert!(spike_count > 0);
}
