// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner sensory execution contract

use sc_neurocore_engine::network_runner::create_population;

#[test]
fn sensory_populations_preserve_finite_public_voltage_contracts() {
    for name in [
        "RetinalGanglion",
        "Merkel",
        "Pacinian",
        "Nociceptor",
        "OlfactoryReceptor",
    ] {
        let mut population =
            create_population(name, 5).unwrap_or_else(|error| panic!("{name}: {error}"));
        for _ in 0..200 {
            population
                .set_currents(&[20.0; 5])
                .expect("matching public current vector must be accepted");
            population.step_all();
        }

        let voltages = population.collect_voltages();
        assert_eq!(voltages.len(), 5, "{name}: voltage count mismatch");
        assert!(voltages.iter().all(|voltage| voltage.is_finite()));
    }
}
