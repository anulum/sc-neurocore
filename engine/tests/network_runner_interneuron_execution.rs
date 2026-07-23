// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner interneuron execution contract

use sc_neurocore_engine::network_runner::{create_population, NetworkRunner, ProjectionRunner};

#[test]
fn interneuron_populations_step_and_reset_through_public_contracts() {
    for name in [
        "PVFastSpiking",
        "SST",
        "VIP",
        "Chandelier",
        "CerebellarBasket",
        "Martinotti",
    ] {
        let mut population =
            create_population(name, 5).unwrap_or_else(|error| panic!("{name}: {error}"));
        for _ in 0..100 {
            population
                .set_currents(&[3.0; 5])
                .expect("matching public current vector must be accepted");
            population.step_all();
        }
        let voltages = population.collect_voltages();
        assert_eq!(voltages.len(), 5, "{name}: voltage count mismatch");
        assert!(voltages.iter().all(|voltage| voltage.is_finite()));

        population.reset_all();
        assert!(population
            .collect_voltages()
            .iter()
            .all(|voltage| voltage.is_finite()));
    }
}

#[test]
fn mixed_interneuron_network_preserves_finite_public_results() {
    let mut runner = NetworkRunner::new();
    let pv = runner.add_population(
        create_population("PVFastSpiking", 3).expect("PV interneuron must remain supported"),
    );
    let sst = runner.add_population(
        create_population("SST", 3).expect("SST interneuron must remain supported"),
    );
    runner
        .step_population_with_currents(pv, &[3.0; 3])
        .expect("matching public current vector must be accepted");
    runner.add_projection(ProjectionRunner::new(
        pv,
        sst,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0; 9],
        0,
    ));

    let results = runner.run(50);

    assert_eq!(results.spike_counts.len(), 2);
    assert_eq!(results.voltages.len(), 2);
    assert!(results
        .voltages
        .iter()
        .flatten()
        .all(|voltage| voltage.is_finite()));
}
