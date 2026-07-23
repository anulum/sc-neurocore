// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner simulation-results public contract

use sc_neurocore_engine::network_runner::{create_population, NetworkRunner, SimResults};

#[test]
fn public_simulation_results_preserve_population_shapes() {
    let mut runner = NetworkRunner::new();
    runner.add_population(
        create_population("Izhikevich", 2).expect("Izhikevich must remain supported"),
    );

    let results: SimResults = runner.run(3);

    assert_eq!(results.spike_counts.len(), 1);
    assert_eq!(results.spike_data.len(), 1);
    assert_eq!(results.voltages.len(), 1);
    assert_eq!(results.voltages[0].len(), 2);
    assert!(results.voltages[0]
        .iter()
        .all(|voltage| voltage.is_finite()));
}
