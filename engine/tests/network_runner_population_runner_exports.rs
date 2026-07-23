// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner population public contract

use sc_neurocore_engine::network_runner::{create_neuron, PopulationRunner};

#[test]
fn public_population_runner_preserves_execution_lifecycle() {
    let neurons = vec![
        create_neuron("Lapicque").expect("Lapicque must remain supported"),
        create_neuron("Lapicque").expect("Lapicque must remain supported"),
    ];
    let mut population = PopulationRunner::new(neurons);

    assert_eq!(population.len(), 2);
    assert!(!population.is_empty());
    population
        .set_currents(&[1.0, 2.0])
        .expect("matching public current vectors must be accepted");
    population.step_all();

    let voltages = population.collect_voltages();
    assert_eq!(voltages.len(), 2);
    assert!(voltages.iter().all(|voltage| voltage.is_finite()));

    population.reset_currents();
    population.reset_all();
    assert!(population
        .collect_voltages()
        .iter()
        .all(|voltage| voltage.is_finite()));
}

#[test]
fn public_population_runner_rejects_mismatched_current_vectors() {
    let mut population = PopulationRunner::new(vec![
        create_neuron("Lapicque").expect("Lapicque must remain supported")
    ]);

    assert_eq!(
        population.set_currents(&[]),
        Err("current vector length mismatch: got 0, expected 1".to_owned())
    );
    assert!(PopulationRunner::new(Vec::new()).is_empty());
}
