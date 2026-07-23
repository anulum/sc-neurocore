// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner execution public contract

use std::panic::{catch_unwind, AssertUnwindSafe};

use sc_neurocore_engine::network_runner::{create_population, NetworkRunner, ProjectionRunner};

#[test]
fn public_network_runner_safely_propagates_from_higher_population_index() {
    let mut runner = NetworkRunner::new();
    let target = runner.add_population(
        create_population("McCullochPittsNeuron", 1)
            .expect("McCulloch-Pitts must remain supported"),
    );
    let source = runner.add_population(
        create_population("McCullochPittsNeuron", 1)
            .expect("McCulloch-Pitts must remain supported"),
    );
    let (source_spikes, _) = runner
        .step_population_with_currents(source, &[1.0])
        .expect("matching current vector must be accepted");
    assert_eq!(source_spikes, vec![1]);
    runner.add_projection(ProjectionRunner::new(
        source,
        target,
        vec![0, 1],
        vec![0],
        vec![1.0],
        0,
    ));

    let results = runner.run(1);

    assert_eq!(results.spike_counts[target], 1);
}

#[test]
fn public_network_runner_rejects_out_of_range_projection_indices() {
    for (source, target, expected_message) in [
        (1, 0, "projection source population index 1 out of range"),
        (0, 1, "projection target population index 1 out of range"),
    ] {
        let mut runner = NetworkRunner::new();
        runner.add_population(
            create_population("McCullochPittsNeuron", 1)
                .expect("McCulloch-Pitts must remain supported"),
        );
        runner.add_projection(ProjectionRunner::new(
            source,
            target,
            vec![0, 1],
            vec![0],
            vec![1.0],
            0,
        ));

        let panic = match catch_unwind(AssertUnwindSafe(|| runner.run(1))) {
            Ok(_) => panic!("invalid projection indices must fail closed"),
            Err(panic) => panic,
        };
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .expect("projection index panic must carry a string message");
        assert_eq!(message, expected_message);
    }
}

#[test]
fn public_network_runner_preserves_self_projection_shapes() {
    let mut runner = NetworkRunner::new();
    let population = runner.add_population(
        create_population("Izhikevich", 4).expect("Izhikevich must remain supported"),
    );
    let mut row_offsets = Vec::new();
    let mut column_indices = Vec::new();
    let mut values = Vec::new();
    for source in 0..4 {
        row_offsets.push(source * 4);
        for target in 0..4 {
            column_indices.push(target);
            values.push(2.0);
        }
    }
    row_offsets.push(16);
    runner.add_projection(ProjectionRunner::new(
        population,
        population,
        row_offsets,
        column_indices,
        values,
        0,
    ));

    let results = runner.run(100);

    assert_eq!(results.spike_counts.len(), 1);
    assert_eq!(results.voltages.len(), 1);
    assert_eq!(results.voltages[0].len(), 4);
}

#[test]
fn public_network_runner_preserves_mixed_population_shapes() {
    let mut runner = NetworkRunner::new();
    let hh = runner.add_population(
        create_population("HodgkinHuxley", 3).expect("Hodgkin-Huxley must remain supported"),
    );
    let adex =
        runner.add_population(create_population("AdEx", 3).expect("AdEx must remain supported"));
    runner
        .step_population_with_currents(hh, &[15.0; 3])
        .expect("matching public current vector must be accepted");
    runner.add_projection(ProjectionRunner::new(
        hh,
        adex,
        vec![0, 3, 6, 9],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![100.0; 9],
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
