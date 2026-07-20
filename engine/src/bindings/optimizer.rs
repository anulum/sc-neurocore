// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SC optimizer PyO3 bindings

//! Python bindings for stochastic-computing hardware design-space optimisation.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Register the SC optimizer functions with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(simulated_annealing_search, module)?)?;
    module.add_function(wrap_pyfunction!(extract_pareto_frontier, module)?)?;
    Ok(())
}

/// Run simulated-annealing design-space search.
#[pyfunction(name = "py_opt_sa_search")]
#[pyo3(signature = (mac_counts, weights, max_luts, max_power, max_latency=0, t_init=1.0, t_min=0.001, alpha=0.95, max_iter=2000, seed=42))]
fn simulated_annealing_search<'py>(
    py: Python<'py>,
    mac_counts: Vec<i64>,
    weights: Vec<f64>,
    max_luts: i64,
    max_power: f64,
    max_latency: i64,
    t_init: f64,
    t_min: f64,
    alpha: f64,
    max_iter: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let candidates: Vec<Vec<crate::optimizer::Candidate>> = mac_counts
        .iter()
        .map(|&mac_count| crate::optimizer::generate_candidates(mac_count))
        .collect();

    let result = crate::optimizer::simulated_annealing(
        &candidates,
        &weights,
        max_luts,
        max_power,
        max_latency,
        t_init,
        t_min,
        alpha,
        max_iter,
        seed,
    );

    let dictionary = PyDict::new(py);
    match result {
        Some(result) => {
            let mut layer_luts = Vec::new();
            let mut layer_power = Vec::new();
            let mut layer_accuracy = Vec::new();
            for (layer_index, &candidate_index) in result.best_config.iter().enumerate() {
                let candidate = &candidates[layer_index][candidate_index];
                layer_luts.push(candidate.luts);
                layer_power.push(candidate.power);
                layer_accuracy.push(candidate.accuracy);
            }

            dictionary.set_item("best_config", result.best_config)?;
            dictionary.set_item("best_score", result.best_score)?;
            dictionary.set_item("pareto_luts", result.pareto_luts)?;
            dictionary.set_item("pareto_power", result.pareto_power)?;
            dictionary.set_item("pareto_score", result.pareto_score)?;
            dictionary.set_item("feasible", true)?;
            dictionary.set_item("layer_luts", layer_luts)?;
            dictionary.set_item("layer_power", layer_power)?;
            dictionary.set_item("layer_accuracy", layer_accuracy)?;
        }
        None => {
            dictionary.set_item("feasible", false)?;
        }
    }
    dictionary.set_item("backend", "rust")?;
    Ok(dictionary.into_any().unbind())
}

/// Extract the non-dominated points from resource and score arrays.
#[pyfunction(name = "py_opt_extract_pareto")]
fn extract_pareto_frontier<'py>(
    py: Python<'py>,
    luts: Vec<i64>,
    power: Vec<f64>,
    score: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    let indices = crate::optimizer::extract_pareto(&luts, &power, &score);
    let dictionary = PyDict::new(py);
    let pareto_luts: Vec<i64> = indices.iter().map(|&index| luts[index]).collect();
    let pareto_power: Vec<f64> = indices.iter().map(|&index| power[index]).collect();
    let pareto_score: Vec<f64> = indices.iter().map(|&index| score[index]).collect();
    dictionary.set_item("indices", indices)?;
    dictionary.set_item("luts", pareto_luts)?;
    dictionary.set_item("power", pareto_power)?;
    dictionary.set_item("score", pareto_score)?;
    dictionary.set_item("backend", "rust")?;
    Ok(dictionary.into_any().unbind())
}
