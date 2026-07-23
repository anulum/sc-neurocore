// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner model-input resilience contract

use sc_neurocore_engine::network_runner::create_neuron;

const MODELS: [&str; 11] = [
    "PVFastSpiking",
    "SST",
    "VIP",
    "Chandelier",
    "CerebellarBasket",
    "Martinotti",
    "RetinalGanglion",
    "Merkel",
    "Pacinian",
    "Nociceptor",
    "OlfactoryReceptor",
];

#[test]
fn reset_restores_finite_state_after_non_finite_input() {
    for name in MODELS {
        let mut neuron = create_neuron(name).unwrap_or_else(|error| panic!("{name}: {error}"));
        for _ in 0..100 {
            neuron.step(2.0);
        }
        for _ in 0..10 {
            neuron.step(f64::NAN);
        }

        neuron.reset();
        assert!(
            neuron.soma_voltage().is_finite(),
            "{name}: reset must restore finite state after NaN input"
        );
    }
}

#[test]
fn reset_restores_finite_state_after_extreme_inputs() {
    for name in MODELS {
        let mut neuron = create_neuron(name).unwrap_or_else(|error| panic!("{name}: {error}"));
        for current in [1e6, -1e6] {
            for _ in 0..50 {
                neuron.step(current);
            }
            neuron.reset();
            assert!(
                neuron.soma_voltage().is_finite(),
                "{name}: reset must restore finite state after {current} input"
            );
        }
    }
}
