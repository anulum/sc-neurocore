// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Population and statistical decoding binding contracts

//! Checked population decoding arrays and admission of actual parallel work.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use super::arrays::{contiguous, matrix_size, readonly, require_prefix};
use crate::{analysis::decoding, runtime_control_binding::ensure_parallel_runtime};

/// Register the five existing population and statistical decoder functions.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_population_vector_decode, module)?)?;
    module.add_function(wrap_pyfunction!(py_bayesian_decode, module)?)?;
    module.add_function(wrap_pyfunction!(py_maximum_likelihood_decode, module)?)?;
    module.add_function(wrap_pyfunction!(py_linear_discriminant_decode, module)?)?;
    module.add_function(wrap_pyfunction!(py_naive_bayes_decode, module)?)?;
    Ok(())
}

/// Decode radians from contiguous int32 spike trains and float64 directions.
/// Use complete windows from the shortest train, preserving missing directions'
/// original zero-radian fallback. Empty bins and zero windows remain lazy.
#[pyfunction]
#[pyo3(signature = (trains, preferred_directions, window=50))]
fn py_population_vector_decode(
    py: Python<'_>,
    trains: Vec<Bound<'_, PyArray1<i32>>>,
    preferred_directions: &Bound<'_, PyArray1<f64>>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let trains: Vec<_> = trains
        .iter()
        .map(|train| readonly(train, "trains"))
        .collect::<PyResult<_>>()?;
    let refs: Vec<_> = trains
        .iter()
        .map(|train| contiguous(train, "trains"))
        .collect::<PyResult<_>>()?;
    let directions = readonly(preferred_directions, "preferred_directions")?;
    let directions = contiguous(&directions, "preferred_directions")?;
    if window > 0 && refs.iter().map(|train| train.len()).min().unwrap_or(0) >= window {
        ensure_parallel_runtime()?;
    }
    Ok(
        decoding::population_vector_decode(&refs, directions, window)
            .into_pyarray(py)
            .into(),
    )
}

fn admit_rates(counts: &[f64], rates: &[f64], stimuli: usize, neurons: usize) -> PyResult<()> {
    if stimuli == 0 || neurons == 0 {
        return Ok(());
    }
    let size = matrix_size(stimuli, neurons)?;
    require_prefix(counts, neurons, "spike_counts")?;
    require_prefix(rates, size, "tuning_rates")?;
    ensure_parallel_runtime()
}

/// Decode a MAP stimulus from float64 counts and row-major tuning rates.
/// Used prefixes contain `n_neurons` and `n_stimuli * n_neurons` elements.
/// Empty prior is uniform; short priors retain the original missing-entry floor.
/// Zero dimensions return zero without starting workers; inputs remain unchanged.
#[pyfunction]
#[pyo3(signature = (spike_counts, tuning_rates, n_stimuli, n_neurons, prior=None))]
fn py_bayesian_decode(
    spike_counts: &Bound<'_, PyArray1<f64>>,
    tuning_rates: &Bound<'_, PyArray1<f64>>,
    n_stimuli: usize,
    n_neurons: usize,
    prior: Option<Bound<'_, PyArray1<f64>>>,
) -> PyResult<usize> {
    let counts = readonly(spike_counts, "spike_counts")?;
    let rates = readonly(tuning_rates, "tuning_rates")?;
    let prior = prior.as_ref().map(|p| readonly(p, "prior")).transpose()?;
    let counts = contiguous(&counts, "spike_counts")?;
    let rates = contiguous(&rates, "tuning_rates")?;
    let prior = prior
        .as_ref()
        .map(|p| contiguous(p, "prior"))
        .transpose()?
        .unwrap_or(&[]);
    admit_rates(counts, rates, n_stimuli, n_neurons)?;
    Ok(decoding::bayesian_decode(
        counts, rates, n_stimuli, n_neurons, prior,
    ))
}

/// Decode maximum likelihood using the Bayesian kernel's uniform-prior path.
/// Validate contiguous float64 buffers and their used prefixes before workers.
/// Zero dimensions and ignored tails preserve the existing numerical contract.
#[pyfunction]
fn py_maximum_likelihood_decode(
    spike_counts: &Bound<'_, PyArray1<f64>>,
    tuning_rates: &Bound<'_, PyArray1<f64>>,
    n_stimuli: usize,
    n_neurons: usize,
) -> PyResult<usize> {
    let counts = readonly(spike_counts, "spike_counts")?;
    let rates = readonly(tuning_rates, "tuning_rates")?;
    let counts = contiguous(&counts, "spike_counts")?;
    let rates = contiguous(&rates, "tuning_rates")?;
    admit_rates(counts, rates, n_stimuli, n_neurons)?;
    Ok(decoding::maximum_likelihood_decode(
        counts, rates, n_stimuli, n_neurons,
    ))
}

/// Return a Fisher discriminant class for flat float64 features and int64 labels.
/// Zero dimensions return zero; a sole used class returns its label without
/// reading feature prefixes or starting workers. Multi-class input and scatter
/// sizes are checked before workers; extra feature and label tails are ignored.
#[pyfunction]
fn py_linear_discriminant_decode(
    train_data: &Bound<'_, PyArray1<f64>>,
    n_samples: usize,
    n_features: usize,
    labels: &Bound<'_, PyArray1<i64>>,
    test_point: &Bound<'_, PyArray1<f64>>,
) -> PyResult<i64> {
    let data = readonly(train_data, "train_data")?;
    let labels = readonly(labels, "labels")?;
    let point = readonly(test_point, "test_point")?;
    let data = contiguous(&data, "train_data")?;
    let labels = contiguous(&labels, "labels")?;
    let point = contiguous(&point, "test_point")?;
    if n_samples == 0 || n_features == 0 {
        return Ok(0);
    }
    require_prefix(labels, n_samples, "labels")?;
    if labels[..n_samples].iter().all(|&label| label == labels[0]) {
        return Ok(labels[0]);
    }
    require_prefix(data, matrix_size(n_samples, n_features)?, "train_data")?;
    require_prefix(point, n_features, "test_point")?;
    matrix_size(n_features, n_features)?;
    ensure_parallel_runtime()?;
    Ok(decoding::linear_discriminant_decode(
        data, n_samples, n_features, labels, point,
    ))
}

/// Return a Gaussian naive Bayes class from contiguous features and int64 labels.
/// Validate every used prefix before the original serial computation.
/// Zero dimensions return zero without workers, including under worker pressure.
#[pyfunction]
fn py_naive_bayes_decode(
    train_data: &Bound<'_, PyArray1<f64>>,
    n_samples: usize,
    n_features: usize,
    labels: &Bound<'_, PyArray1<i64>>,
    test_point: &Bound<'_, PyArray1<f64>>,
) -> PyResult<i64> {
    let data = readonly(train_data, "train_data")?;
    let labels = readonly(labels, "labels")?;
    let point = readonly(test_point, "test_point")?;
    let data = contiguous(&data, "train_data")?;
    let labels = contiguous(&labels, "labels")?;
    let point = contiguous(&point, "test_point")?;
    if n_samples == 0 || n_features == 0 {
        return Ok(0);
    }
    require_prefix(labels, n_samples, "labels")?;
    require_prefix(data, matrix_size(n_samples, n_features)?, "train_data")?;
    require_prefix(point, n_features, "test_point")?;
    Ok(decoding::naive_bayes_decode(
        data, n_samples, n_features, labels, point,
    ))
}
