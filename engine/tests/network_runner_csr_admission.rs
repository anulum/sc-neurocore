// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner CSR projection admission contract

//! `admit_projection` refuses malformed CSR connectivity before allocating and
//! leaves the network unchanged, while well-formed forward, reverse, self and
//! delayed projections deliver exactly the events `add_projection` delivers.

use sc_neurocore_engine::network_runner::{
    admit_csr, create_population, NetworkRunner, ProjectionRunner,
};

/// Return a runner with `sizes.len()` McCulloch-Pitts populations.
fn runner_with(sizes: &[usize]) -> NetworkRunner {
    let mut runner = NetworkRunner::new();
    for &n in sizes {
        runner.add_population(
            create_population("McCullochPittsNeuron", n)
                .expect("McCulloch-Pitts must remain supported"),
        );
    }
    runner
}

#[test]
fn well_formed_csr_is_admitted_with_unsigned_storage() {
    let (offsets, columns) = admit_csr(2, 3, &[0, 2, 3], &[0, 2, 1], &[1.0, 2.0, 3.0])
        .expect("well-formed CSR must be admitted");
    assert_eq!(offsets, vec![0, 2, 3]);
    assert_eq!(columns, vec![0, 2, 1]);
    // Unused trailing storage is allowed, as in scipy; it is never read.
    assert!(admit_csr(1, 1, &[0, 1], &[0, -7], &[1.0, 9.0]).is_ok());
    // An empty source population has the single offset 0.
    assert!(admit_csr(0, 1, &[0], &[], &[]).is_ok());
}

#[test]
fn every_malformed_csr_is_refused_with_its_violation() {
    let cases: [(&[i64], &[i64], &[f64], &str); 9] = [
        (&[], &[], &[], "row_offsets has 0 entries"),
        (&[0, 1], &[0], &[1.0], "row_offsets has 2 entries"),
        (&[0, 1, 1, 1], &[0], &[1.0], "row_offsets has 4 entries"),
        (&[1, 1, 1], &[0], &[1.0], "must start at 0"),
        (&[0, 2, 1], &[0, 0], &[1.0, 1.0], "must not decrease: entry 2"),
        (&[0, -1, 0], &[], &[], "must not decrease: entry 1"),
        (&[0, 1, 2], &[0, 0], &[1.0], "col_indices has 2 entries but values has 1"),
        (&[0, 1, 3], &[0, 0], &[1.0, 1.0], "the last row offset 3 exceeds"),
        (&[0, 1, 2], &[0, 3], &[1.0, 1.0], "col_indices[1] = 3 is outside"),
    ];
    for (offsets, columns, values, expected) in cases {
        let message = admit_csr(2, 3, offsets, columns, values)
            .expect_err("malformed CSR must be refused");
        assert!(message.contains(expected), "{message:?} lacks {expected:?}");
    }
    let negative = admit_csr(1, 3, &[0, 1], &[-1], &[1.0]).expect_err("negative column");
    assert!(negative.contains("col_indices[0] = -1 is outside"), "{negative}");
}

#[test]
fn a_refused_projection_leaves_the_network_unchanged() {
    let mut runner = runner_with(&[1, 1]);
    runner
        .admit_projection(0, 1, &[0, 1], &[0], vec![1.0], 0)
        .expect("healthy projection");
    for (src, tgt, offsets, columns, expected) in [
        (2usize, 0usize, vec![0i64, 1], vec![0i64], "source population index 2 out of range"),
        (0, 5, vec![0, 1], vec![0], "target population index 5 out of range"),
        (0, 1, vec![0, 1], vec![1], "col_indices[0] = 1 is outside"),
        (0, 1, vec![], vec![], "row_offsets has 0 entries"),
    ] {
        let message = runner
            .admit_projection(src, tgt, &offsets, &columns, vec![1.0; columns.len()], 3)
            .expect_err("malformed projection must be refused");
        assert!(message.contains(expected), "{message:?} lacks {expected:?}");
        assert_eq!(runner.projections.len(), 1);
    }
    runner
        .step_population_with_currents(0, &[1.0])
        .expect("matching current vector");
    assert_eq!(runner.run(1).spike_counts, vec![0, 1]);
}

/// Return the target spike counts of `steps` steps after one source spike.
fn deliver(admitted: bool, src: usize, tgt: usize, delay: usize, steps: usize) -> Vec<usize> {
    let mut runner = runner_with(&[1, 1]);
    if admitted {
        runner
            .admit_projection(src, tgt, &[0, 1], &[0], vec![1.0], delay)
            .expect("healthy projection");
    } else {
        runner.add_projection(ProjectionRunner::new(
            src,
            tgt,
            vec![0, 1],
            vec![0],
            vec![1.0],
            delay,
        ));
    }
    runner
        .step_population_with_currents(src, &[1.0])
        .expect("matching current vector");
    (0..steps).map(|_| runner.run(1).spike_counts[tgt]).collect()
}

#[test]
fn admitted_forward_reverse_self_and_delayed_projections_deliver_as_before() {
    for (src, tgt, delay) in [(0, 1, 0), (1, 0, 0), (0, 0, 0), (0, 1, 2)] {
        let admitted = deliver(true, src, tgt, delay, 4);
        assert_eq!(admitted, deliver(false, src, tgt, delay, 4));
        assert!(admitted.contains(&1), "({src}, {tgt}, {delay}) delivers {admitted:?}");
    }
    assert_eq!(deliver(true, 0, 1, 2, 4), vec![0, 0, 1, 0]);
}
