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
