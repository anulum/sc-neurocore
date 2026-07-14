// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Spike-train analysis PyO3 bindings

//! Python bindings for spike-train and neural-signal analysis.

use crate::analysis;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Register the analysis functions on the Python extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Analysis functions (P0-A: spike_stats)
    m.add_function(wrap_pyfunction!(py_spike_times, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi, m)?)?;
    m.add_function(wrap_pyfunction!(py_firing_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_count, m)?)?;
    m.add_function(wrap_pyfunction!(py_bin_spike_train, m)?)?;
    m.add_function(wrap_pyfunction!(py_instantaneous_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_psth, m)?)?;
    m.add_function(wrap_pyfunction!(py_cv_isi, m)?)?;
    m.add_function(wrap_pyfunction!(py_cv2, m)?)?;
    m.add_function(wrap_pyfunction!(py_local_variation, m)?)?;
    m.add_function(wrap_pyfunction!(py_fano_factor, m)?)?;
    m.add_function(wrap_pyfunction!(py_lempel_ziv_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(py_permutation_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_hurst_exponent, m)?)?;
    m.add_function(wrap_pyfunction!(py_approximate_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_entropy, m)?)?;
    // correlation
    m.add_function(wrap_pyfunction!(py_cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_pairwise_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_event_synchronization, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_train_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_time_tiling_coefficient, m)?)?;
    m.add_function(wrap_pyfunction!(py_covariance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_autocorrelation_time, m)?)?;
    m.add_function(wrap_pyfunction!(py_noise_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_signal_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_count_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(py_joint_psth, m)?)?;
    m.add_function(wrap_pyfunction!(py_coincidence_index, m)?)?;
    // distance
    m.add_function(wrap_pyfunction!(py_van_rossum_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_victor_purpura_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_sync, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_sync_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_profile, m)?)?;
    m.add_function(wrap_pyfunction!(py_adaptive_spike_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_schreiber_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_hunter_milton_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(py_earth_movers_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_multi_neuron_victor_purpura, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_distance_matrix, m)?)?;
    // information
    m.add_function(wrap_pyfunction!(py_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(py_transfer_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_train_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_noise_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(py_stimulus_specific_information, m)?)?;
    m.add_function(wrap_pyfunction!(py_kozachenko_leonenko_mi, m)?)?;
    // causality
    m.add_function(wrap_pyfunction!(py_pairwise_granger_causality, m)?)?;
    m.add_function(wrap_pyfunction!(py_conditional_granger_causality, m)?)?;
    m.add_function(wrap_pyfunction!(py_spectral_granger_causality, m)?)?;
    m.add_function(wrap_pyfunction!(py_partial_directed_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(py_directed_transfer_function, m)?)?;
    // decoding
    m.add_function(wrap_pyfunction!(py_population_vector_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_bayesian_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_maximum_likelihood_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_linear_discriminant_decode, m)?)?;
    m.add_function(wrap_pyfunction!(py_naive_bayes_decode, m)?)?;
    // neural_decoders (P1)
    m.add_function(wrap_pyfunction!(py_tokenise_spikes, m)?)?;
    m.add_function(wrap_pyfunction!(py_sinusoidal_position_encode, m)?)?;
    m.add_function(wrap_pyfunction!(py_scaled_dot_product_attention, m)?)?;
    m.add_function(wrap_pyfunction!(py_gaussian_attention, m)?)?;
    m.add_function(wrap_pyfunction!(py_infonce_loss, m)?)?;
    // network
    m.add_function(wrap_pyfunction!(py_functional_connectivity, m)?)?;
    m.add_function(wrap_pyfunction!(py_unitary_events, m)?)?;
    m.add_function(wrap_pyfunction!(py_cell_assembly_detection, m)?)?;
    m.add_function(wrap_pyfunction!(py_synfire_chain_detection, m)?)?;
    // surrogates
    m.add_function(wrap_pyfunction!(py_surrogate_isi_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(py_surrogate_dither, m)?)?;
    m.add_function(wrap_pyfunction!(py_homogeneous_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(py_gamma_process, m)?)?;
    m.add_function(wrap_pyfunction!(py_compound_poisson_process, m)?)?;
    m.add_function(wrap_pyfunction!(py_surrogate_joint_isi, m)?)?;
    m.add_function(wrap_pyfunction!(py_surrogate_bin_shuffling, m)?)?;
    m.add_function(wrap_pyfunction!(py_surrogate_spike_train_shifting, m)?)?;
    // temporal
    m.add_function(wrap_pyfunction!(py_burst_detection, m)?)?;
    m.add_function(wrap_pyfunction!(py_first_spike_latency, m)?)?;
    m.add_function(wrap_pyfunction!(py_response_onset, m)?)?;
    m.add_function(wrap_pyfunction!(py_change_point_detection, m)?)?;
    // patterns
    m.add_function(wrap_pyfunction!(py_spike_directionality, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_train_order, m)?)?;
    m.add_function(wrap_pyfunction!(py_cubic_higher_order, m)?)?;
    // spectral
    m.add_function(wrap_pyfunction!(py_power_spectrum, m)?)?;
    // waveform
    m.add_function(wrap_pyfunction!(py_waveform_width, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_amplitude, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_repolarization_slope, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_recovery_slope, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_halfwidth, m)?)?;
    m.add_function(wrap_pyfunction!(py_waveform_pt_ratio, m)?)?;
    // point_process
    m.add_function(wrap_pyfunction!(py_conditional_intensity, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_hazard_function, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_survivor_function, m)?)?;
    m.add_function(wrap_pyfunction!(py_renewal_density, m)?)?;
    // stimulus
    m.add_function(wrap_pyfunction!(py_spike_triggered_average, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_triggered_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(py_spatial_information, m)?)?;
    m.add_function(wrap_pyfunction!(py_place_field_detection, m)?)?;
    m.add_function(wrap_pyfunction!(py_tuning_curve, m)?)?;
    // lfp
    m.add_function(wrap_pyfunction!(py_phase_locking_value, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_field_coherence, m)?)?;
    m.add_function(wrap_pyfunction!(py_spike_phase_histogram, m)?)?;
    // sorting_quality
    m.add_function(wrap_pyfunction!(py_isolation_distance, m)?)?;
    m.add_function(wrap_pyfunction!(py_l_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_silhouette_score, m)?)?;
    m.add_function(wrap_pyfunction!(py_d_prime, m)?)?;
    m.add_function(wrap_pyfunction!(py_isi_violation_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_presence_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_amplitude_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(py_snr, m)?)?;
    m.add_function(wrap_pyfunction!(py_nn_hit_rate, m)?)?;
    m.add_function(wrap_pyfunction!(py_drift_metric, m)?)?;
    // dimensionality
    m.add_function(wrap_pyfunction!(py_spike_train_pca, m)?)?;
    m.add_function(wrap_pyfunction!(py_demixed_pca, m)?)?;
    m.add_function(wrap_pyfunction!(py_factor_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(py_pca_components, m)?)?;
    m.add_function(wrap_pyfunction!(py_demixed_components, m)?)?;
    m.add_function(wrap_pyfunction!(py_factor_loadings, m)?)?;
    // gpfa
    m.add_function(wrap_pyfunction!(py_gpfa, m)?)?;
    m.add_function(wrap_pyfunction!(py_gpfa_em, m)?)?;
    m.add_function(wrap_pyfunction!(py_gpfa_transform, m)?)?;
    // spade
    m.add_function(wrap_pyfunction!(py_spade_detect, m)?)?;
    Ok(())
}

// ── Analysis PyO3 wrappers (P0-A: spike_stats) ─────────────────────

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_spike_times(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
) -> Py<PyArray1<f64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::spike_times(data, dt)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_isi(py: Python<'_>, binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> Py<PyArray1<f64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::isi(data, dt).into_pyarray(py).into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_firing_rate(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::firing_rate(data, dt)
}

#[pyfunction]
fn py_spike_count(binary_train: PyReadonlyArray1<'_, i32>) -> i64 {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::spike_count(data)
}

#[pyfunction]
#[pyo3(signature = (binary_train, bin_size=10))]
fn py_bin_spike_train(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
) -> Py<PyArray1<i64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::basic::bin_spike_train(data, bin_size)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, kernel="gaussian", sigma_ms=10.0))]
fn py_instantaneous_rate(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, f64>,
    dt: f64,
    kernel: &str,
    sigma_ms: f64,
) -> Py<PyArray1<f64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::rate::instantaneous_rate(data, dt, kernel, sigma_ms)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (trials, bin_ms=10.0, dt=0.001))]
fn py_psth(
    py: Python<'_>,
    trials: Vec<PyReadonlyArray1<'_, f64>>,
    bin_ms: f64,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<f64>> = trials
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let (rates, centers) = analysis::rate::psth(&vecs, bin_ms, dt);
    (
        rates.into_pyarray(py).into(),
        centers.into_pyarray(py).into(),
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_cv_isi(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::cv_isi(data, dt)
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_cv2(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::cv2(data, dt)
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_local_variation(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::local_variation(data, dt)
}

#[pyfunction]
#[pyo3(signature = (binary_train, window_ms=50.0, dt=0.001))]
fn py_fano_factor(binary_train: PyReadonlyArray1<'_, i32>, window_ms: f64, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::fano_factor(data, window_ms, dt)
}

#[pyfunction]
fn py_lempel_ziv_complexity(binary_train: PyReadonlyArray1<'_, i32>) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::lempel_ziv_complexity(data)
}

#[pyfunction]
#[pyo3(signature = (binary_train, order=3, delay=1))]
fn py_permutation_entropy(
    binary_train: PyReadonlyArray1<'_, i32>,
    order: usize,
    delay: usize,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::permutation_entropy(data, order, delay)
}

#[pyfunction]
#[pyo3(signature = (binary_train, min_window=10))]
fn py_hurst_exponent(binary_train: PyReadonlyArray1<'_, i32>, min_window: usize) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::hurst_exponent(data, min_window)
}

#[pyfunction]
#[pyo3(signature = (binary_train, m=2, r_factor=0.2))]
fn py_approximate_entropy(binary_train: PyReadonlyArray1<'_, i32>, m: usize, r_factor: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::approximate_entropy(data, m, r_factor)
}

#[pyfunction]
#[pyo3(signature = (binary_train, m=2, r_factor=0.2))]
fn py_sample_entropy(binary_train: PyReadonlyArray1<'_, i32>, m: usize, r_factor: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::variability::sample_entropy(data, m, r_factor)
}

// ── Correlation PyO3 wrappers (P0-A: spike_stats/correlation) ────

#[pyfunction]
#[pyo3(signature = (train_a, train_b, max_lag_ms=50.0, dt=0.001))]
fn py_cross_correlation(
    py: Python<'_>,
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    max_lag_ms: f64,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    let (cc, lags) = analysis::correlation::cross_correlation(a, b, max_lag_ms, dt);
    (cc.into_pyarray(py).into(), lags.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (trains, dt=0.001))]
fn py_pairwise_correlation(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    dt: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::pairwise_correlation(&refs, dt);
    let n = mat.len();
    let flat: Vec<f64> = mat.into_iter().flatten().collect();
    numpy::PyArray2::from_vec2(py, &flat.chunks(n).map(|c| c.to_vec()).collect::<Vec<_>>())
        .unwrap()
        .into()
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, tau_ms=5.0))]
fn py_event_synchronization(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    tau_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::correlation::event_synchronization(a, b, dt, tau_ms)
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001))]
fn py_spike_train_coherence(
    py: Python<'_>,
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    let (coh, freqs) = analysis::correlation::spike_train_coherence(a, b, dt);
    (coh.into_pyarray(py).into(), freqs.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, delta_ms=5.0))]
fn py_spike_time_tiling_coefficient(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    delta_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::correlation::spike_time_tiling_coefficient(a, b, dt, delta_ms)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=10))]
fn py_covariance_matrix(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::covariance_matrix(&refs, bin_size);
    let n = mat.len();
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, max_lag_ms=100.0))]
fn py_autocorrelation_time(
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    max_lag_ms: f64,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::correlation::autocorrelation_time(data, dt, max_lag_ms)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=50))]
fn py_noise_correlation(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::noise_correlation(&refs, bin_size);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=50))]
fn py_signal_correlation(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::signal_correlation(&refs, bin_size);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (trains, window=50))]
fn py_spike_count_covariance(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    window: usize,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::correlation::spike_count_covariance(&refs, window);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, bin_size=10))]
fn py_joint_psth(
    py: Python<'_>,
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
) -> Py<PyArray2<f64>> {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    let (flat, n) = analysis::correlation::joint_psth(a, b, bin_size);
    if n == 0 {
        return numpy::PyArray2::zeros(py, [0, 0], false).into();
    }
    let rows: Vec<Vec<f64>> = flat.chunks(n).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows).unwrap().into()
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, delta_ms=2.0))]
fn py_coincidence_index(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    delta_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::correlation::coincidence_index(a, b, dt, delta_ms)
}

// ── Distance PyO3 wrappers (P0-A: spike_stats/distance) ─────────

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, tau_ms=10.0))]
fn py_van_rossum_distance(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    tau_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::distance::van_rossum_distance(a, b, dt, tau_ms)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, cost_per_s=1000.0))]
fn py_victor_purpura_distance(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    cost_per_s: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::victor_purpura_distance(a, b, cost_per_s)
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001))]
fn py_isi_distance(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::distance::isi_distance(a, b, dt)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0))]
fn py_spike_distance(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::spike_distance(a, b, t_start, t_end)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0))]
fn py_spike_sync(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::spike_sync(a, b, t_start, t_end)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, n_bins=50, t_start=0.0, t_end=1.0))]
fn py_spike_sync_profile(
    py: Python<'_>,
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    t_start: f64,
    t_end: f64,
) -> Py<PyArray1<f64>> {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::spike_sync_profile(a, b, n_bins, t_start, t_end)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, n_bins=50, t_start=0.0, t_end=1.0))]
fn py_spike_profile(
    py: Python<'_>,
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    t_start: f64,
    t_end: f64,
) -> Py<PyArray1<f64>> {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::spike_profile(a, b, n_bins, t_start, t_end)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train_a, binary_train_b, dt=0.001, n_bins=50))]
fn py_isi_profile(
    py: Python<'_>,
    binary_train_a: PyReadonlyArray1<'_, i32>,
    binary_train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    n_bins: usize,
) -> Py<PyArray1<f64>> {
    let a = binary_train_a.as_slice().unwrap();
    let b = binary_train_b.as_slice().unwrap();
    analysis::distance::isi_profile(a, b, dt, n_bins)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0, cost=0.0))]
fn py_adaptive_spike_distance(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
    cost: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::adaptive_spike_distance(a, b, t_start, t_end, cost)
}

#[pyfunction]
#[pyo3(signature = (train_a, train_b, dt=0.001, sigma_ms=5.0))]
fn py_schreiber_similarity(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    dt: f64,
    sigma_ms: f64,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::distance::schreiber_similarity(a, b, dt, sigma_ms)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, dt_max=0.01))]
fn py_hunter_milton_similarity(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    dt_max: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::hunter_milton_similarity(a, b, dt_max)
}

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0, n_bins=100))]
fn py_earth_movers_distance(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
    n_bins: usize,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::distance::earth_movers_distance(a, b, t_start, t_end, n_bins)
}

#[pyfunction]
#[pyo3(signature = (spike_times_list, cost_per_s=1000.0))]
fn py_multi_neuron_victor_purpura(
    py: Python<'_>,
    spike_times_list: Vec<PyReadonlyArray1<'_, f64>>,
    cost_per_s: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<f64>> = spike_times_list
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::distance::multi_neuron_victor_purpura(&refs, cost_per_s);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (spike_times_list, metric="spike_distance", t_start=0.0, t_end=1.0))]
fn py_spike_distance_matrix(
    py: Python<'_>,
    spike_times_list: Vec<PyReadonlyArray1<'_, f64>>,
    metric: &str,
    t_start: f64,
    t_end: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<f64>> = spike_times_list
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::distance::spike_distance_matrix(&refs, metric, t_start, t_end);
    let rows: Vec<Vec<f64>> = mat.into_iter().collect();
    let n = rows.len();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

// ── Information PyO3 wrappers (P0-A: spike_stats/information) ────

#[pyfunction]
#[pyo3(signature = (train_a, train_b, bin_size=10))]
fn py_mutual_information(
    train_a: PyReadonlyArray1<'_, i32>,
    train_b: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
) -> f64 {
    let a = train_a.as_slice().unwrap();
    let b = train_b.as_slice().unwrap();
    analysis::information::mutual_information(a, b, bin_size)
}

#[pyfunction]
#[pyo3(signature = (source, target, bin_size=10, lag=1))]
fn py_transfer_entropy(
    source: PyReadonlyArray1<'_, i32>,
    target: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    lag: usize,
) -> f64 {
    let s = source.as_slice().unwrap();
    let t = target.as_slice().unwrap();
    analysis::information::transfer_entropy(s, t, bin_size, lag)
}

#[pyfunction]
#[pyo3(signature = (binary_train, bin_size=10, word_length=4))]
fn py_spike_train_entropy(
    binary_train: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    word_length: usize,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::information::spike_train_entropy(data, bin_size, word_length)
}

#[pyfunction]
#[pyo3(signature = (binary_train, n_trials=10, bin_size=10, word_length=4))]
fn py_noise_entropy(
    binary_train: PyReadonlyArray1<'_, i32>,
    n_trials: usize,
    bin_size: usize,
    word_length: usize,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::information::noise_entropy(data, n_trials, bin_size, word_length)
}

#[pyfunction]
fn py_stimulus_specific_information(
    spike_counts: PyReadonlyArray1<'_, f64>,
    stimulus_ids: PyReadonlyArray1<'_, i64>,
) -> f64 {
    let counts = spike_counts.as_slice().unwrap();
    let ids = stimulus_ids.as_slice().unwrap();
    analysis::information::stimulus_specific_information(counts, ids)
}

#[pyfunction]
#[pyo3(signature = (x, y, k=3))]
fn py_kozachenko_leonenko_mi(
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    k: usize,
) -> f64 {
    let xd = x.as_slice().unwrap();
    let yd = y.as_slice().unwrap();
    analysis::information::kozachenko_leonenko_mi(xd, yd, k)
}

// ── Causality PyO3 wrappers (P0-A: spike_stats/causality) ───────

#[pyfunction]
#[pyo3(signature = (source, target, bin_size=10, order=5))]
fn py_pairwise_granger_causality(
    source: PyReadonlyArray1<'_, i32>,
    target: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    order: usize,
) -> f64 {
    let s = source.as_slice().unwrap();
    let t = target.as_slice().unwrap();
    analysis::causality::pairwise_granger_causality(s, t, bin_size, order)
}

#[pyfunction]
#[pyo3(signature = (source, target, condition, bin_size=10, order=5))]
fn py_conditional_granger_causality(
    source: PyReadonlyArray1<'_, i32>,
    target: PyReadonlyArray1<'_, i32>,
    condition: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    order: usize,
) -> f64 {
    let s = source.as_slice().unwrap();
    let t = target.as_slice().unwrap();
    let c = condition.as_slice().unwrap();
    analysis::causality::conditional_granger_causality(s, t, c, bin_size, order)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_spectral_granger_causality(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Py<PyArray1<f64>>, usize, usize) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (gc, d) = analysis::causality::spectral_granger_causality(&refs, bin_size, order, n_freqs);
    (gc.into_pyarray(py).into(), d, n_freqs)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_partial_directed_coherence(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Py<PyArray1<f64>>, usize, usize) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (pdc, d) = analysis::causality::partial_directed_coherence(&refs, bin_size, order, n_freqs);
    (pdc.into_pyarray(py).into(), d, n_freqs)
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=10, order=5, n_freqs=64))]
fn py_directed_transfer_function(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    order: usize,
    n_freqs: usize,
) -> (Py<PyArray1<f64>>, usize, usize) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (dtf, d) = analysis::causality::directed_transfer_function(&refs, bin_size, order, n_freqs);
    (dtf.into_pyarray(py).into(), d, n_freqs)
}

// ── Decoding PyO3 wrappers (P0-A: spike_stats/decoding) ─────────

#[pyfunction]
#[pyo3(signature = (trains, preferred_directions, window=50))]
fn py_population_vector_decode(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    preferred_directions: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> Py<PyArray1<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let dirs = preferred_directions.as_slice().unwrap();
    analysis::decoding::population_vector_decode(&refs, dirs, window)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (spike_counts, tuning_rates, n_stimuli, n_neurons, prior=None))]
fn py_bayesian_decode(
    spike_counts: PyReadonlyArray1<'_, f64>,
    tuning_rates: PyReadonlyArray1<'_, f64>,
    n_stimuli: usize,
    n_neurons: usize,
    prior: Option<PyReadonlyArray1<'_, f64>>,
) -> usize {
    let counts = spike_counts.as_slice().unwrap();
    let rates = tuning_rates.as_slice().unwrap();
    let p: Vec<f64> = prior
        .map(|p| p.as_slice().unwrap().to_vec())
        .unwrap_or_default();
    analysis::decoding::bayesian_decode(counts, rates, n_stimuli, n_neurons, &p)
}

#[pyfunction]
fn py_maximum_likelihood_decode(
    spike_counts: PyReadonlyArray1<'_, f64>,
    tuning_rates: PyReadonlyArray1<'_, f64>,
    n_stimuli: usize,
    n_neurons: usize,
) -> usize {
    let counts = spike_counts.as_slice().unwrap();
    let rates = tuning_rates.as_slice().unwrap();
    analysis::decoding::maximum_likelihood_decode(counts, rates, n_stimuli, n_neurons)
}

#[pyfunction]
fn py_linear_discriminant_decode(
    train_data: PyReadonlyArray1<'_, f64>,
    n_samples: usize,
    n_features: usize,
    labels: PyReadonlyArray1<'_, i64>,
    test_point: PyReadonlyArray1<'_, f64>,
) -> i64 {
    let data = train_data.as_slice().unwrap();
    let lbl = labels.as_slice().unwrap();
    let tp = test_point.as_slice().unwrap();
    analysis::decoding::linear_discriminant_decode(data, n_samples, n_features, lbl, tp)
}

#[pyfunction]
fn py_naive_bayes_decode(
    train_data: PyReadonlyArray1<'_, f64>,
    n_samples: usize,
    n_features: usize,
    labels: PyReadonlyArray1<'_, i64>,
    test_point: PyReadonlyArray1<'_, f64>,
) -> i64 {
    let data = train_data.as_slice().unwrap();
    let lbl = labels.as_slice().unwrap();
    let tp = test_point.as_slice().unwrap();
    analysis::decoding::naive_bayes_decode(data, n_samples, n_features, lbl, tp)
}

// ── Neural decoder PyO3 wrappers (P1: neural_decoders) ─────────

#[pyfunction]
#[pyo3(signature = (trains, dt=1.0))]
fn py_tokenise_spikes(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    dt: f64,
) -> (Py<PyArray1<i64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let tokens = analysis::neural_decoders::tokenise_spikes(&refs, dt);
    let uids: Vec<i64> = tokens.iter().map(|t| t.0 as i64).collect();
    let times: Vec<f64> = tokens.iter().map(|t| t.1).collect();
    (uids.into_pyarray(py).into(), times.into_pyarray(py).into())
}

#[pyfunction]
fn py_sinusoidal_position_encode(
    py: Python<'_>,
    timestamps: PyReadonlyArray1<'_, f64>,
    d_model: usize,
) -> Py<PyArray1<f64>> {
    let ts = timestamps.as_slice().unwrap();
    analysis::neural_decoders::sinusoidal_position_encode(ts, d_model)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
fn py_scaled_dot_product_attention(
    py: Python<'_>,
    queries: PyReadonlyArray1<'_, f64>,
    keys: PyReadonlyArray1<'_, f64>,
    values: PyReadonlyArray1<'_, f64>,
    nq: usize,
    nk: usize,
    d: usize,
) -> Py<PyArray1<f64>> {
    let q = queries.as_slice().unwrap();
    let k = keys.as_slice().unwrap();
    let v = values.as_slice().unwrap();
    analysis::neural_decoders::scaled_dot_product_attention(q, k, v, nq, nk, d)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
fn py_gaussian_attention(
    py: Python<'_>,
    queries: PyReadonlyArray1<'_, f64>,
    keys: PyReadonlyArray1<'_, f64>,
    values: PyReadonlyArray1<'_, f64>,
    nq: usize,
    nk: usize,
    d: usize,
    sigma: f64,
) -> Py<PyArray1<f64>> {
    let q = queries.as_slice().unwrap();
    let k = keys.as_slice().unwrap();
    let v = values.as_slice().unwrap();
    analysis::neural_decoders::gaussian_attention(q, k, v, nq, nk, d, sigma)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
fn py_infonce_loss(
    anchors: PyReadonlyArray1<'_, f64>,
    positives: PyReadonlyArray1<'_, f64>,
    n: usize,
    d: usize,
    temperature: f64,
) -> f64 {
    let a = anchors.as_slice().unwrap();
    let p = positives.as_slice().unwrap();
    analysis::neural_decoders::infonce_loss(a, p, n, d, temperature)
}

// ── Network PyO3 wrappers (P0-A: spike_stats/network) ───────────

#[pyfunction]
#[pyo3(signature = (trains, max_lag_ms=20.0, dt=0.001))]
fn py_functional_connectivity(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    max_lag_ms: f64,
    dt: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::network::functional_connectivity(&refs, max_lag_ms, dt);
    let n = refs.len();
    let rows: Vec<Vec<f64>> = mat.chunks(n).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=5, alpha=0.05))]
fn py_unitary_events(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    alpha: f64,
) -> Py<PyArray1<i64>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let result = analysis::network::unitary_events(&refs, bin_size, alpha);
    let as_i64: Vec<i64> = result.into_iter().map(|v| v as i64).collect();
    as_i64.into_pyarray(py).into()
}

#[pyfunction]
#[pyo3(signature = (trains, bin_size=5, threshold=2.0))]
fn py_cell_assembly_detection(
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    bin_size: usize,
    threshold: f64,
) -> Vec<Vec<usize>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    analysis::network::cell_assembly_detection(&refs, bin_size, threshold)
}

#[pyfunction]
#[pyo3(signature = (trains, dt=0.001, max_delay_ms=20.0, min_chain_length=3))]
fn py_synfire_chain_detection(
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    dt: f64,
    max_delay_ms: f64,
    min_chain_length: usize,
) -> Vec<Vec<usize>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    analysis::network::synfire_chain_detection(&refs, dt, max_delay_ms, min_chain_length)
}

// ── Surrogates PyO3 wrappers (P0-A: spike_stats/surrogates) ─────

#[pyfunction]
#[pyo3(signature = (binary_train, seed=0))]
fn py_surrogate_isi_shuffle(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_isi_shuffle(data, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dither_ms=5.0, dt=0.001, seed=0))]
fn py_surrogate_dither(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dither_ms: f64,
    dt: f64,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_dither(data, dither_ms, dt, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (rate_hz, duration_s, dt=0.001, seed=0))]
fn py_homogeneous_poisson(
    py: Python<'_>,
    rate_hz: f64,
    duration_s: f64,
    dt: f64,
    seed: u64,
) -> Py<PyArray1<f64>> {
    analysis::surrogates::homogeneous_poisson(rate_hz, duration_s, dt, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (rate_hz, shape, duration_s, dt=0.001, seed=0))]
fn py_gamma_process(
    py: Python<'_>,
    rate_hz: f64,
    shape: f64,
    duration_s: f64,
    dt: f64,
    seed: u64,
) -> Py<PyArray1<f64>> {
    analysis::surrogates::gamma_process(rate_hz, shape, duration_s, dt, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (rate_hz, burst_mean, duration_s, dt=0.001, seed=0))]
fn py_compound_poisson_process(
    py: Python<'_>,
    rate_hz: f64,
    burst_mean: f64,
    duration_s: f64,
    dt: f64,
    seed: u64,
) -> Py<PyArray1<f64>> {
    analysis::surrogates::compound_poisson_process(rate_hz, burst_mean, duration_s, dt, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, seed=0))]
fn py_surrogate_joint_isi(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_joint_isi(data, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, bin_size=10, seed=0))]
fn py_surrogate_bin_shuffling(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_bin_shuffling(data, bin_size, seed)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, max_shift=50, seed=0))]
fn py_surrogate_spike_train_shifting(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    max_shift: usize,
    seed: u64,
) -> Py<PyArray1<i32>> {
    let data = binary_train.as_slice().unwrap();
    analysis::surrogates::surrogate_spike_train_shifting(data, max_shift, seed)
        .into_pyarray(py)
        .into()
}

// ── Temporal PyO3 wrappers (P0-A: spike_stats/temporal) ─────────

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, max_isi_ms=10.0, min_spikes=3))]
fn py_burst_detection(
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    max_isi_ms: f64,
    min_spikes: usize,
) -> Vec<(f64, f64, usize)> {
    let data = binary_train.as_slice().unwrap();
    analysis::temporal::burst_detection(data, dt, max_isi_ms, min_spikes)
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_first_spike_latency(binary_train: PyReadonlyArray1<'_, i32>, dt: f64) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::temporal::first_spike_latency(data, dt)
}

#[pyfunction]
#[pyo3(signature = (binary_train, baseline_steps=100, dt=0.001, threshold_sigma=3.0))]
fn py_response_onset(
    binary_train: PyReadonlyArray1<'_, i32>,
    baseline_steps: usize,
    dt: f64,
    threshold_sigma: f64,
) -> f64 {
    let data = binary_train.as_slice().unwrap();
    analysis::temporal::response_onset(data, baseline_steps, dt, threshold_sigma)
}

#[pyfunction]
#[pyo3(signature = (binary_train, bin_size=50, threshold=3.0))]
fn py_change_point_detection(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    bin_size: usize,
    threshold: f64,
) -> Py<PyArray1<i64>> {
    let data = binary_train.as_slice().unwrap();
    let cps = analysis::temporal::change_point_detection(data, bin_size, threshold);
    let as_i64: Vec<i64> = cps.into_iter().map(|v| v as i64).collect();
    as_i64.into_pyarray(py).into()
}

// ── Patterns PyO3 wrappers (P0-A: spike_stats/patterns) ─────────

#[pyfunction]
#[pyo3(signature = (times_a, times_b, t_start=0.0, t_end=1.0))]
fn py_spike_directionality(
    times_a: PyReadonlyArray1<'_, f64>,
    times_b: PyReadonlyArray1<'_, f64>,
    t_start: f64,
    t_end: f64,
) -> f64 {
    let a = times_a.as_slice().unwrap();
    let b = times_b.as_slice().unwrap();
    analysis::patterns::spike_directionality(a, b, t_start, t_end)
}

#[pyfunction]
#[pyo3(signature = (times_list, t_start=0.0, t_end=1.0))]
fn py_spike_train_order(
    py: Python<'_>,
    times_list: Vec<PyReadonlyArray1<'_, f64>>,
    t_start: f64,
    t_end: f64,
) -> Py<PyArray2<f64>> {
    let vecs: Vec<Vec<f64>> = times_list
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[f64]> = vecs.iter().map(|v| v.as_slice()).collect();
    let mat = analysis::patterns::spike_train_order(&refs, t_start, t_end);
    let n = refs.len();
    let rows: Vec<Vec<f64>> = mat.chunks(n).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [n, n], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, max_lag=20))]
fn py_cubic_higher_order(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    max_lag: usize,
) -> Py<PyArray2<f64>> {
    let data = binary_train.as_slice().unwrap();
    let c3 = analysis::patterns::cubic_higher_order(data, dt, max_lag);
    let rows: Vec<Vec<f64>> = c3.chunks(max_lag).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [max_lag, max_lag], false))
        .into()
}

// ── Spectral PyO3 wrappers (P0-A: spike_stats/spectral) ─────────

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001))]
fn py_power_spectrum(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let data = binary_train.as_slice().unwrap();
    let (psd, freqs) = analysis::spectral::power_spectrum(data, dt);
    (psd.into_pyarray(py).into(), freqs.into_pyarray(py).into())
}

// ── Waveform PyO3 wrappers (P0-A: spike_stats/waveform) ─────────

#[pyfunction]
#[pyo3(signature = (waveform, dt=3.3333333333333335e-05))]
fn py_waveform_width(waveform: PyReadonlyArray1<'_, f64>, dt: f64) -> f64 {
    analysis::waveform::waveform_width(waveform.as_slice().unwrap(), dt)
}

#[pyfunction]
fn py_waveform_amplitude(waveform: PyReadonlyArray1<'_, f64>) -> f64 {
    analysis::waveform::waveform_amplitude(waveform.as_slice().unwrap())
}

#[pyfunction]
#[pyo3(signature = (waveform, dt=3.3333333333333335e-05))]
fn py_waveform_repolarization_slope(waveform: PyReadonlyArray1<'_, f64>, dt: f64) -> f64 {
    analysis::waveform::waveform_repolarization_slope(waveform.as_slice().unwrap(), dt)
}

#[pyfunction]
#[pyo3(signature = (waveform, dt=3.3333333333333335e-05))]
fn py_waveform_recovery_slope(waveform: PyReadonlyArray1<'_, f64>, dt: f64) -> f64 {
    analysis::waveform::waveform_recovery_slope(waveform.as_slice().unwrap(), dt)
}

#[pyfunction]
#[pyo3(signature = (waveform, dt=3.3333333333333335e-05))]
fn py_waveform_halfwidth(waveform: PyReadonlyArray1<'_, f64>, dt: f64) -> f64 {
    analysis::waveform::waveform_halfwidth(waveform.as_slice().unwrap(), dt)
}

#[pyfunction]
fn py_waveform_pt_ratio(waveform: PyReadonlyArray1<'_, f64>) -> f64 {
    analysis::waveform::waveform_pt_ratio(waveform.as_slice().unwrap())
}

// ── Point process PyO3 wrappers (P0-A: spike_stats/point_process) ──

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, window_ms=50.0))]
fn py_conditional_intensity(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    window_ms: f64,
) -> Py<PyArray1<f64>> {
    let data = binary_train.as_slice().unwrap();
    analysis::point_process::conditional_intensity(data, dt, window_ms)
        .into_pyarray(py)
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, bins=30))]
fn py_isi_hazard_function(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    bins: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let data = binary_train.as_slice().unwrap();
    let (hazard, centres) = analysis::point_process::isi_hazard_function(data, dt, bins);
    (
        hazard.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, bins=30))]
fn py_isi_survivor_function(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    bins: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let data = binary_train.as_slice().unwrap();
    let (surv, centres) = analysis::point_process::isi_survivor_function(data, dt, bins);
    (
        surv.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, bins=30))]
fn py_renewal_density(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    bins: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let data = binary_train.as_slice().unwrap();
    let (dens, centres) = analysis::point_process::renewal_density(data, dt, bins);
    (
        dens.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

// ── Stimulus PyO3 wrappers (P0-A: spike_stats/stimulus) ─────────

#[pyfunction]
#[pyo3(signature = (stimulus, binary_train, window_steps=50))]
fn py_spike_triggered_average(
    py: Python<'_>,
    stimulus: PyReadonlyArray1<'_, f64>,
    binary_train: PyReadonlyArray1<'_, i32>,
    window_steps: usize,
) -> Py<PyArray1<f64>> {
    analysis::stimulus::spike_triggered_average(
        stimulus.as_slice().unwrap(),
        binary_train.as_slice().unwrap(),
        window_steps,
    )
    .into_pyarray(py)
    .into()
}

#[pyfunction]
#[pyo3(signature = (stimulus, binary_train, window_steps=50))]
fn py_spike_triggered_covariance(
    py: Python<'_>,
    stimulus: PyReadonlyArray1<'_, f64>,
    binary_train: PyReadonlyArray1<'_, i32>,
    window_steps: usize,
) -> Py<PyArray2<f64>> {
    let cov = analysis::stimulus::spike_triggered_covariance(
        stimulus.as_slice().unwrap(),
        binary_train.as_slice().unwrap(),
        window_steps,
    );
    let rows: Vec<Vec<f64>> = cov.chunks(window_steps).map(|c| c.to_vec()).collect();
    numpy::PyArray2::from_vec2(py, &rows)
        .unwrap_or_else(|_| numpy::PyArray2::zeros(py, [window_steps, window_steps], false))
        .into()
}

#[pyfunction]
#[pyo3(signature = (binary_train, positions, n_bins=20, dt=0.001))]
fn py_spatial_information(
    binary_train: PyReadonlyArray1<'_, i32>,
    positions: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    dt: f64,
) -> f64 {
    analysis::stimulus::spatial_information(
        binary_train.as_slice().unwrap(),
        positions.as_slice().unwrap(),
        n_bins,
        dt,
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, positions, n_bins=50, threshold_std=2.0, dt=0.001))]
fn py_place_field_detection(
    binary_train: PyReadonlyArray1<'_, i32>,
    positions: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    threshold_std: f64,
    dt: f64,
) -> Vec<(f64, f64)> {
    analysis::stimulus::place_field_detection(
        binary_train.as_slice().unwrap(),
        positions.as_slice().unwrap(),
        n_bins,
        threshold_std,
        dt,
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, stimulus_values, n_bins=20, dt=0.001))]
fn py_tuning_curve(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    stimulus_values: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let (rates, centres) = analysis::stimulus::tuning_curve(
        binary_train.as_slice().unwrap(),
        stimulus_values.as_slice().unwrap(),
        n_bins,
        dt,
    );
    (
        rates.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

// ── LFP PyO3 wrappers (P0-A: spike_stats/lfp) ─────────────────

#[pyfunction]
fn py_phase_locking_value(
    binary_train: PyReadonlyArray1<'_, i32>,
    lfp_signal: PyReadonlyArray1<'_, f64>,
) -> f64 {
    analysis::lfp::phase_locking_value(
        binary_train.as_slice().unwrap(),
        lfp_signal.as_slice().unwrap(),
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, lfp_signal, dt=0.001))]
fn py_spike_field_coherence(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    lfp_signal: PyReadonlyArray1<'_, f64>,
    dt: f64,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let (sfc, freqs) = analysis::lfp::spike_field_coherence(
        binary_train.as_slice().unwrap(),
        lfp_signal.as_slice().unwrap(),
        dt,
    );
    (sfc.into_pyarray(py).into(), freqs.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (binary_train, lfp_signal, n_bins=36))]
fn py_spike_phase_histogram(
    py: Python<'_>,
    binary_train: PyReadonlyArray1<'_, i32>,
    lfp_signal: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> (Py<PyArray1<i64>>, Py<PyArray1<f64>>) {
    let (hist, centres) = analysis::lfp::spike_phase_histogram(
        binary_train.as_slice().unwrap(),
        lfp_signal.as_slice().unwrap(),
        n_bins,
    );
    (
        hist.into_pyarray(py).into(),
        centres.into_pyarray(py).into(),
    )
}

// ── Sorting quality PyO3 wrappers (P0-A: spike_stats/sorting_quality)

#[pyfunction]
fn py_isolation_distance(
    cluster: PyReadonlyArray2<'_, f64>,
    noise: PyReadonlyArray2<'_, f64>,
) -> f64 {
    let c_shape = cluster.shape();
    let n_shape = noise.shape();
    let d = c_shape[1];
    let c_data: Vec<f64> = cluster.as_slice().unwrap().to_vec();
    let n_data: Vec<f64> = noise.as_slice().unwrap().to_vec();
    analysis::sorting_quality::isolation_distance(&c_data, c_shape[0], &n_data, n_shape[0], d)
}

#[pyfunction]
fn py_l_ratio(cluster: PyReadonlyArray2<'_, f64>, noise: PyReadonlyArray2<'_, f64>) -> f64 {
    let c_shape = cluster.shape();
    let n_shape = noise.shape();
    let d = c_shape[1];
    let c_data: Vec<f64> = cluster.as_slice().unwrap().to_vec();
    let n_data: Vec<f64> = noise.as_slice().unwrap().to_vec();
    analysis::sorting_quality::l_ratio(&c_data, c_shape[0], &n_data, n_shape[0], d)
}

#[pyfunction]
fn py_silhouette_score(
    features: PyReadonlyArray2<'_, f64>,
    labels: PyReadonlyArray1<'_, i64>,
) -> f64 {
    let shape = features.shape();
    let f_data: Vec<f64> = features.as_slice().unwrap().to_vec();
    let l_data: Vec<i64> = labels.as_slice().unwrap().to_vec();
    analysis::sorting_quality::silhouette_score(&f_data, shape[0], shape[1], &l_data)
}

#[pyfunction]
fn py_d_prime(cluster_a: PyReadonlyArray2<'_, f64>, cluster_b: PyReadonlyArray2<'_, f64>) -> f64 {
    let a_shape = cluster_a.shape();
    let b_shape = cluster_b.shape();
    let d = a_shape[1];
    let a_data: Vec<f64> = cluster_a.as_slice().unwrap().to_vec();
    let b_data: Vec<f64> = cluster_b.as_slice().unwrap().to_vec();
    analysis::sorting_quality::d_prime(&a_data, a_shape[0], &b_data, b_shape[0], d)
}

#[pyfunction]
#[pyo3(signature = (binary_train, dt=0.001, refractory_ms=1.5))]
fn py_isi_violation_rate(
    binary_train: PyReadonlyArray1<'_, i32>,
    dt: f64,
    refractory_ms: f64,
) -> f64 {
    analysis::sorting_quality::isi_violation_rate(
        binary_train.as_slice().unwrap(),
        dt,
        refractory_ms,
    )
}

#[pyfunction]
#[pyo3(signature = (binary_train, n_bins=100))]
fn py_presence_ratio(binary_train: PyReadonlyArray1<'_, i32>, n_bins: usize) -> f64 {
    analysis::sorting_quality::presence_ratio(binary_train.as_slice().unwrap(), n_bins)
}

#[pyfunction]
#[pyo3(signature = (amplitudes, bins=100))]
fn py_amplitude_cutoff(amplitudes: PyReadonlyArray1<'_, f64>, bins: usize) -> f64 {
    analysis::sorting_quality::amplitude_cutoff(amplitudes.as_slice().unwrap(), bins)
}

#[pyfunction]
fn py_snr(waveforms: PyReadonlyArray2<'_, f64>) -> f64 {
    let shape = waveforms.shape();
    let data: Vec<f64> = waveforms.as_slice().unwrap().to_vec();
    analysis::sorting_quality::snr(&data, shape[0], shape[1])
}

#[pyfunction]
#[pyo3(signature = (cluster, noise, k=4))]
fn py_nn_hit_rate(
    cluster: PyReadonlyArray2<'_, f64>,
    noise: PyReadonlyArray2<'_, f64>,
    k: usize,
) -> f64 {
    let c_shape = cluster.shape();
    let n_shape = noise.shape();
    let d = c_shape[1];
    let c_data: Vec<f64> = cluster.as_slice().unwrap().to_vec();
    let n_data: Vec<f64> = noise.as_slice().unwrap().to_vec();
    analysis::sorting_quality::nn_hit_rate(&c_data, c_shape[0], &n_data, n_shape[0], d, k)
}

#[pyfunction]
#[pyo3(signature = (waveforms, timestamps, n_bins=10))]
fn py_drift_metric(
    waveforms: PyReadonlyArray2<'_, f64>,
    timestamps: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> f64 {
    let shape = waveforms.shape();
    let data: Vec<f64> = waveforms.as_slice().unwrap().to_vec();
    let ts: Vec<f64> = timestamps.as_slice().unwrap().to_vec();
    analysis::sorting_quality::drift_metric(&data, shape[0], shape[1], &ts, n_bins)
}

// ── Dimensionality PyO3 wrappers (P0-A: spike_stats/dimensionality)

#[pyfunction]
#[pyo3(signature = (trains, n_components=3, bin_size=10))]
fn py_spike_train_pca(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    n_components: usize,
    bin_size: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (proj, expl) = analysis::dimensionality::spike_train_pca(&refs, n_components, bin_size);
    (proj.into_pyarray(py).into(), expl.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (conditions, n_components=3, bin_size=10))]
fn py_demixed_pca(
    py: Python<'_>,
    conditions: Vec<Vec<PyReadonlyArray1<'_, i32>>>,
    n_components: usize,
    bin_size: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<Vec<i32>>> = conditions
        .iter()
        .map(|cond| {
            cond.iter()
                .map(|t| t.as_slice().unwrap().to_vec())
                .collect()
        })
        .collect();
    let refs: Vec<Vec<&[i32]>> = vecs
        .iter()
        .map(|cond| cond.iter().map(|v| v.as_slice()).collect())
        .collect();
    let (proj, expl) = analysis::dimensionality::demixed_pca(&refs, n_components, bin_size);
    (proj.into_pyarray(py).into(), expl.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (trains, n_factors=3, bin_size=10, n_iter=50))]
fn py_factor_analysis(
    py: Python<'_>,
    trains: Vec<PyReadonlyArray1<'_, i32>>,
    n_factors: usize,
    bin_size: usize,
    n_iter: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let (loadings, psi) =
        analysis::dimensionality::factor_analysis(&refs, n_factors, bin_size, n_iter);
    (
        loadings.into_pyarray(py).into(),
        psi.into_pyarray(py).into(),
    )
}

// Matrix-input wrappers: the caller bins and mean-centres once, so every backend
// (NumPy / Rust / Julia / Go / Mojo) shares an identical input matrix and the
// outputs agree to floating-point round-off.

#[pyfunction]
#[pyo3(signature = (mat, n_components=3))]
fn py_pca_components(
    py: Python<'_>,
    mat: PyReadonlyArray2<'_, f64>,
    n_components: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let shape = mat.shape();
    let data: Vec<f64> = mat.as_slice().unwrap().to_vec();
    let (proj, expl) =
        analysis::dimensionality::pca_from_centered(&data, shape[0], shape[1], n_components);
    (proj.into_pyarray(py).into(), expl.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (mean_mat, n_components=3))]
fn py_demixed_components(
    py: Python<'_>,
    mean_mat: PyReadonlyArray2<'_, f64>,
    n_components: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let shape = mean_mat.shape();
    let data: Vec<f64> = mean_mat.as_slice().unwrap().to_vec();
    let (proj, expl) =
        analysis::dimensionality::demixed_from_centered(&data, shape[0], shape[1], n_components);
    (proj.into_pyarray(py).into(), expl.into_pyarray(py).into())
}

#[pyfunction]
#[pyo3(signature = (mat, n_factors=3, n_iter=50))]
fn py_factor_loadings(
    py: Python<'_>,
    mat: PyReadonlyArray2<'_, f64>,
    n_factors: usize,
    n_iter: usize,
) -> (Py<PyArray1<f64>>, Py<PyArray1<f64>>) {
    let shape = mat.shape();
    let data: Vec<f64> = mat.as_slice().unwrap().to_vec();
    let (loadings, psi) =
        analysis::dimensionality::fa_from_centered(&data, shape[0], shape[1], n_factors, n_iter);
    (
        loadings.into_pyarray(py).into(),
        psi.into_pyarray(py).into(),
    )
}

// ── GPFA PyO3 wrappers (P0-A: spike_stats/gpfa) ─────────────────

#[pyfunction]
#[pyo3(signature = (trains, n_latents=3, bin_ms=20.0, dt=0.001, max_iter=50, tol=1e-4, seed=42))]
fn py_gpfa<'py>(
    py: Python<'py>,
    trains: Vec<PyReadonlyArray1<'py, i32>>,
    n_latents: usize,
    bin_ms: f64,
    dt: f64,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let result = analysis::gpfa::gpfa(&refs, n_latents, bin_ms, dt, max_iter, tol, seed);

    let dict = PyDict::new(py);
    dict.set_item("trajectories", result.trajectories.into_pyarray(py))?;
    dict.set_item("C", result.c.into_pyarray(py))?;
    dict.set_item("d", result.d.into_pyarray(py))?;
    dict.set_item("R", result.r.into_pyarray(py))?;
    dict.set_item("tau", result.tau.into_pyarray(py))?;
    dict.set_item("log_likelihoods", result.log_likelihoods.into_pyarray(py))?;
    dict.set_item("n_latents", result.n_latents)?;
    dict.set_item("n_bins", result.n_bins)?;
    dict.set_item("n_neurons", result.n_neurons)?;
    Ok(dict)
}

/// Run the GPFA EM loop from a caller-supplied deterministic initialisation.
///
/// Parity contract with `sc_neurocore.analysis.spike_stats.gpfa.gpfa_em`: identical
/// inputs (the PCA init computed once in Python) produce the same trajectories,
/// parameters and exact-marginal log-likelihoods up to floating-point round-off.
#[pyfunction]
#[pyo3(signature = (y, n_neurons, n_bins, c0, d0, r0_diag, tau, n_latents, max_iter, tol))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_gpfa_em<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    n_neurons: usize,
    n_bins: usize,
    c0: PyReadonlyArray1<'py, f64>,
    d0: PyReadonlyArray1<'py, f64>,
    r0_diag: PyReadonlyArray1<'py, f64>,
    tau: PyReadonlyArray1<'py, f64>,
    n_latents: usize,
    max_iter: usize,
    tol: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Vec<f64>,
)> {
    let (x_post, c, d, r, log_liks) = analysis::gpfa::gpfa_em_from_init(
        y.as_slice()?,
        c0.as_slice()?,
        d0.as_slice()?,
        r0_diag.as_slice()?,
        tau.as_slice()?,
        n_neurons,
        n_bins,
        n_latents,
        max_iter,
        tol,
    );
    Ok((
        x_post.into_pyarray(py),
        c.into_pyarray(py),
        d.into_pyarray(py),
        r.into_pyarray(py),
        log_liks,
    ))
}

#[pyfunction]
#[pyo3(signature = (new_trains, c, d, r, tau, n_latents, bin_ms=20.0, dt=0.001))]
fn py_gpfa_transform(
    py: Python<'_>,
    new_trains: Vec<PyReadonlyArray1<'_, i32>>,
    c: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray1<'_, f64>,
    r: PyReadonlyArray1<'_, f64>,
    tau: PyReadonlyArray1<'_, f64>,
    n_latents: usize,
    bin_ms: f64,
    dt: f64,
) -> Py<PyArray1<f64>> {
    let vecs: Vec<Vec<i32>> = new_trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let proj = analysis::gpfa::gpfa_transform(
        &refs,
        c.as_slice().unwrap(),
        d.as_slice().unwrap(),
        r.as_slice().unwrap(),
        tau.as_slice().unwrap(),
        n_latents,
        bin_ms,
        dt,
    );
    proj.into_pyarray(py).into()
}

// ── SPADE PyO3 wrappers (P0-A: spike_stats/spade) ─────────────

#[pyfunction]
#[pyo3(signature = (trains, bin_ms=5.0, dt=0.001, min_support=3, max_pattern_size=5, n_surrogates=100, alpha=0.05, seed=42))]
fn py_spade_detect<'py>(
    py: Python<'py>,
    trains: Vec<PyReadonlyArray1<'py, i32>>,
    bin_ms: f64,
    dt: f64,
    min_support: usize,
    max_pattern_size: usize,
    n_surrogates: usize,
    alpha: f64,
    seed: u64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let vecs: Vec<Vec<i32>> = trains
        .iter()
        .map(|t| t.as_slice().unwrap().to_vec())
        .collect();
    let refs: Vec<&[i32]> = vecs.iter().map(|v| v.as_slice()).collect();
    let results = analysis::spade::spade_detect(
        &refs,
        bin_ms,
        dt,
        min_support,
        max_pattern_size,
        n_surrogates,
        alpha,
        seed,
    );
    let mut dicts = Vec::new();
    for pat in results {
        let dict = PyDict::new(py);
        dict.set_item(
            "neurons",
            pat.neurons.iter().map(|&n| n as i64).collect::<Vec<_>>(),
        )?;
        dict.set_item("lags", pat.lags.clone())?;
        dict.set_item("count", pat.count as i64)?;
        dict.set_item("p_value", pat.p_value)?;
        dicts.push(dict);
    }
    Ok(dicts)
}
