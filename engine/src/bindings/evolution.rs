// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Evolutionary substrate PyO3 bindings

//! Python bindings for population-level evolutionary operators.

use pyo3::prelude::*;

/// Register the evolutionary-substrate functions with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(batch_mutate_weights, module)?)?;
    module.add_function(wrap_pyfunction!(batch_evaluate_fitness, module)?)?;
    module.add_function(wrap_pyfunction!(batch_crossover, module)?)?;
    module.add_function(wrap_pyfunction!(population_diversity, module)?)?;
    module.add_function(wrap_pyfunction!(novelty_scores, module)?)?;
    module.add_function(wrap_pyfunction!(tournament_select, module)?)?;
    Ok(())
}

/// Batch-mutate population weights.
#[pyfunction(name = "py_evo_batch_mutate")]
#[pyo3(signature = (genomes, mutation_rate=0.1, mutation_scale=0.1, seed=42))]
fn batch_mutate_weights(
    _py: Python<'_>,
    mut genomes: Vec<Vec<f64>>,
    mutation_rate: f64,
    mutation_scale: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    crate::evo::batch_mutate_weights(&mut genomes, mutation_rate, mutation_scale, seed);
    genomes
}

/// Evaluate population fitness in one Rust call.
#[pyfunction(name = "py_evo_batch_fitness")]
fn batch_evaluate_fitness(
    _py: Python<'_>,
    genomes: Vec<Vec<f64>>,
    inputs: Vec<f64>,
    target: f64,
) -> Vec<f64> {
    crate::evo::batch_evaluate_fitness(&genomes, &inputs, target)
}

/// Apply uniform crossover to paired parent populations.
#[pyfunction(name = "py_evo_batch_crossover")]
#[pyo3(signature = (parents_a, parents_b, seed=42))]
fn batch_crossover(
    _py: Python<'_>,
    parents_a: Vec<Vec<f64>>,
    parents_b: Vec<Vec<f64>>,
    seed: u64,
) -> Vec<Vec<f64>> {
    crate::evo::batch_crossover(&parents_a, &parents_b, seed)
}

/// Return mean pairwise L2 distance for a population.
#[pyfunction(name = "py_evo_diversity")]
fn population_diversity(_py: Python<'_>, genomes: Vec<Vec<f64>>) -> f64 {
    crate::evo::population_diversity(&genomes)
}

/// Score population novelty against an archive.
#[pyfunction(name = "py_evo_novelty")]
#[pyo3(signature = (genomes, archive, k_nearest=5))]
fn novelty_scores(
    _py: Python<'_>,
    genomes: Vec<Vec<f64>>,
    archive: Vec<Vec<f64>>,
    k_nearest: usize,
) -> Vec<f64> {
    crate::evo::novelty_scores(&genomes, &archive, k_nearest)
}

/// Select population indices by seeded tournaments.
#[pyfunction(name = "py_evo_tournament")]
#[pyo3(signature = (fitness, n_select, tournament_size=3, seed=42))]
fn tournament_select(
    _py: Python<'_>,
    fitness: Vec<f64>,
    n_select: usize,
    tournament_size: usize,
    seed: u64,
) -> Vec<usize> {
    crate::evo::tournament_select(&fitness, n_select, tournament_size, seed)
}
