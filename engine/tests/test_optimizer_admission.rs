// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Public optimizer search admission

//! Reject missing search dimensions without changing admitted seeded results.

use sc_neurocore_engine::optimizer::{
    generate_candidates, simulated_annealing, Candidate, SAResult,
};

fn search(candidates: &[Vec<Candidate>], weights: &[f64], iterations: usize) -> Option<SAResult> {
    simulated_annealing(
        candidates, weights, 100_000, 1000.0, 0, 1.0, 0.001, 0.95, iterations, 42,
    )
}

#[test]
fn empty_search_is_refused_even_without_iterations() {
    for iterations in [0, 10] {
        assert!(search(&[], &[], iterations).is_none());
    }
}

#[test]
fn missing_layer_weight_is_refused_before_scoring() {
    let candidates = vec![generate_candidates(10), generate_candidates(20)];
    for weights in [&[][..], &[1.0][..]] {
        for iterations in [0, 10] {
            assert!(search(&candidates, weights, iterations).is_none());
        }
    }
}

#[test]
fn empty_candidate_layer_is_refused_before_indexing() {
    for candidates in [vec![vec![]], vec![generate_candidates(10), vec![]]] {
        assert!(search(&candidates, &[1.0, 1.0], 10).is_none());
    }
}

#[test]
fn admitted_search_preserves_seeded_configuration_and_score() {
    let candidates = vec![generate_candidates(10), generate_candidates(20)];
    let first = search(&candidates, &[1.0, 1.0], 10).unwrap();
    assert_eq!(first.best_config, [29, 14]);
    assert_eq!(first.best_score, 0.99736328125);
    let second = search(&candidates, &[1.0, 1.0], 10).unwrap();
    assert_eq!(first.best_config, second.best_config);
    assert_eq!(first.pareto_luts, second.pareto_luts);
    assert_eq!(first.pareto_power, second.pareto_power);
    assert_eq!(first.pareto_score, second.pareto_score);
    let trailing = search(&candidates, &[1.0, 1.0, 2.0], 0).unwrap();
    let initial = search(&candidates, &[1.0, 1.0], 0).unwrap();
    assert_eq!(trailing.best_config, initial.best_config);
    assert_eq!(trailing.best_score, initial.best_score / 2.0);
}
