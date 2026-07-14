// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Quantum annealing PyO3 bindings

//! Python bindings for the quantum annealing acceleration primitives.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::quantum;

/// Compute Ising energy for a spin configuration.
///
/// Args:
///     h_indices, h_values: linear biases (parallel arrays)
///     j_i, j_j, j_values: quadratic couplings (parallel arrays)
///     spins: spin configuration (+1/-1)
///     offset: constant energy offset
#[pyfunction]
#[pyo3(signature = (h_indices, h_values, j_i, j_j, j_values, spins, offset=0.0))]
fn py_qa_ising_energy(
    _py: Python<'_>,
    h_indices: Vec<usize>,
    h_values: Vec<f64>,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    spins: Vec<i8>,
    offset: f64,
) -> f64 {
    let h: Vec<(usize, f64)> = h_indices.into_iter().zip(h_values).collect();
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();
    quantum::ising_energy(&h, &j, &spins, offset)
}

/// Batch compute Ising energies for many configurations.
#[pyfunction]
#[pyo3(signature = (h_indices, h_values, j_i, j_j, j_values, configs, offset=0.0))]
fn py_qa_batch_ising_energy(
    _py: Python<'_>,
    h_indices: Vec<usize>,
    h_values: Vec<f64>,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    configs: Vec<Vec<i8>>,
    offset: f64,
) -> Vec<f64> {
    let h: Vec<(usize, f64)> = h_indices.into_iter().zip(h_values).collect();
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();
    quantum::batch_ising_energy(&h, &j, &configs, offset)
}

/// Run simulated annealing on an Ising model (Rust-accelerated).
///
/// Returns dict with best_spins, best_energy, energies, samples.
#[pyfunction]
#[pyo3(signature = (h_indices, h_values, j_i, j_j, j_values, n_qubits, offset=0.0, n_sweeps=1000, num_reads=10, beta_start=0.1, beta_end=10.0, seed=42))]
fn py_qa_simulated_annealing<'py>(
    py: Python<'py>,
    h_indices: Vec<usize>,
    h_values: Vec<f64>,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    n_qubits: usize,
    offset: f64,
    n_sweeps: usize,
    num_reads: usize,
    beta_start: f64,
    beta_end: f64,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let h: Vec<(usize, f64)> = h_indices.into_iter().zip(h_values).collect();
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();

    let (best_spins, best_energy, all_energies, all_samples) = quantum::simulated_annealing(
        &h, &j, n_qubits, offset, n_sweeps, num_reads, beta_start, beta_end, seed,
    );

    let dict = PyDict::new(py);
    dict.set_item("best_spins", best_spins)?;
    dict.set_item("best_energy", best_energy)?;
    dict.set_item("energies", all_energies)?;
    dict.set_item("samples", all_samples)?;
    dict.set_item("n_sweeps", n_sweeps)?;
    dict.set_item("num_reads", num_reads)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Apply gauge transform to Ising biases and couplings.
#[pyfunction]
#[allow(clippy::type_complexity)] // tuple shape mirrors Python's QUBO format
fn py_qa_gauge_transform(
    _py: Python<'_>,
    h_indices: Vec<usize>,
    h_values: Vec<f64>,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    gauge: Vec<i8>,
) -> (Vec<(usize, f64)>, Vec<((usize, usize), f64)>) {
    let h: Vec<(usize, f64)> = h_indices.into_iter().zip(h_values).collect();
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();
    quantum::gauge_transform(&h, &j, &gauge)
}

/// Generate random gauge vectors.
#[pyfunction]
#[pyo3(signature = (n_qubits, n_gauges=10, seed=42))]
fn py_qa_generate_gauges(
    _py: Python<'_>,
    n_qubits: usize,
    n_gauges: usize,
    seed: u64,
) -> Vec<Vec<i8>> {
    quantum::generate_gauges(n_qubits, n_gauges, seed)
}

/// Greedy graph partitioning for problem decomposition.
#[pyfunction]
#[pyo3(signature = (n_qubits, j_i, j_j, j_values, max_partition_size=64))]
fn py_qa_greedy_partition(
    _py: Python<'_>,
    n_qubits: usize,
    j_i: Vec<usize>,
    j_j: Vec<usize>,
    j_values: Vec<f64>,
    max_partition_size: usize,
) -> Vec<Vec<usize>> {
    let j: Vec<((usize, usize), f64)> = j_i.into_iter().zip(j_j).zip(j_values).collect();
    quantum::greedy_partition(n_qubits, &j, max_partition_size)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_qa_ising_energy, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_batch_ising_energy, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_simulated_annealing, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_gauge_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_generate_gauges, m)?)?;
    m.add_function(wrap_pyfunction!(py_qa_greedy_partition, m)?)?;
    Ok(())
}
