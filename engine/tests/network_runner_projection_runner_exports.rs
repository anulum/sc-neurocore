// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner projection public contract

use sc_neurocore_engine::network_runner::ProjectionRunner;

#[test]
fn public_projection_runner_preserves_csr_scatter() {
    let mut projection = ProjectionRunner::new(
        0,
        1,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![5.0, 3.0, 2.0, 4.0],
        0,
    );
    let mut target_currents = vec![0.0; 2];

    projection.propagate(&[1, 0], &mut target_currents);

    assert_eq!(projection.src_pop, 0);
    assert_eq!(projection.tgt_pop, 1);
    assert_eq!(target_currents, vec![5.0, 3.0]);
}

#[test]
fn public_projection_runner_preserves_discrete_delay() {
    let mut projection = ProjectionRunner::new(0, 1, vec![0, 1], vec![0], vec![2.5], 2);
    let mut target_currents = vec![0.0];

    projection.propagate(&[1], &mut target_currents);
    assert_eq!(target_currents, vec![0.0]);
    projection.propagate(&[0], &mut target_currents);
    assert_eq!(target_currents, vec![0.0]);
    projection.propagate(&[0], &mut target_currents);
    assert_eq!(target_currents, vec![2.5]);
}
