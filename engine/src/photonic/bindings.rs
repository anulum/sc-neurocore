// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Photonic NoC PyO3 bindings

//! Python bindings for the photonic routing, MZI, crosstalk, and power-budget core.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::photonic;

// ── Photonic NoC PyO3 Wrappers ───────────────────────────────────────

/// Route waveguides on a mesh topology (Rust-accelerated).
#[pyfunction]
#[pyo3(signature = (adjacency_flat, n, pitch_um=250.0, loss_db_per_cm=2.0))]
fn py_ph_route_waveguides<'py>(
    py: Python<'py>,
    adjacency_flat: Vec<f64>,
    n: usize,
    pitch_um: f64,
    loss_db_per_cm: f64,
) -> PyResult<Py<PyAny>> {
    let result = photonic::route_waveguides(&adjacency_flat, n, pitch_um, loss_db_per_cm);

    let dict = PyDict::new(py);
    let sources: Vec<usize> = result.iter().map(|r| r.source).collect();
    let targets: Vec<usize> = result.iter().map(|r| r.target).collect();
    let lengths: Vec<f64> = result.iter().map(|r| r.length_um).collect();
    let losses: Vec<f64> = result.iter().map(|r| r.loss_db).collect();
    let crossings: Vec<usize> = result.iter().map(|r| r.n_crossings).collect();

    dict.set_item("sources", sources)?;
    dict.set_item("targets", targets)?;
    dict.set_item("lengths_um", lengths)?;
    dict.set_item("losses_db", losses)?;
    dict.set_item("crossings", crossings)?;
    dict.set_item("n_segments", result.len())?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Compute MZI 2×2 transfer matrix for a given phase.
#[pyfunction]
fn py_ph_mzi_transfer_matrix(_py: Python<'_>, phase_rad: f64) -> Vec<f64> {
    photonic::mzi_transfer_matrix(phase_rad).to_vec()
}

/// Cascade multiple MZI stages via matrix multiplication.
#[pyfunction]
fn py_ph_cascade_mzi(_py: Python<'_>, phases: Vec<f64>) -> Vec<f64> {
    photonic::cascade_mzi(&phases).to_vec()
}

/// Analyze WDM crosstalk (Rust-accelerated).
#[pyfunction]
#[pyo3(signature = (channel_ids, wavelengths, bandwidths, powers, adjacent_xt_db=-25.0))]
fn py_ph_analyze_crosstalk<'py>(
    py: Python<'py>,
    channel_ids: Vec<usize>,
    wavelengths: Vec<f64>,
    bandwidths: Vec<f64>,
    powers: Vec<f64>,
    adjacent_xt_db: f64,
) -> PyResult<Py<PyAny>> {
    let channels: Vec<(usize, f64, f64, f64)> = channel_ids
        .into_iter()
        .zip(wavelengths)
        .zip(bandwidths)
        .zip(powers)
        .map(|(((id, wl), bw), p)| (id, wl, bw, p))
        .collect();

    let result = photonic::analyze_crosstalk(&channels, adjacent_xt_db);

    let dict = PyDict::new(py);
    let ids: Vec<usize> = result.iter().map(|r| r.channel_id).collect();
    let xts: Vec<f64> = result.iter().map(|r| r.crosstalk_db).collect();
    let osnrs: Vec<f64> = result.iter().map(|r| r.osnr_db).collect();
    let adjs: Vec<usize> = result.iter().map(|r| r.n_adjacent).collect();

    dict.set_item("channel_ids", ids)?;
    dict.set_item("crosstalk_db", xts)?;
    dict.set_item("osnr_db", osnrs)?;
    dict.set_item("n_adjacent", adjs)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Analyze optical power budget (Rust-accelerated).
#[pyfunction]
#[pyo3(signature = (wg_sources, wg_targets, wg_losses, laser_power_dbm=0.0, detector_sensitivity_dbm=-20.0))]
fn py_ph_analyze_power_budget<'py>(
    py: Python<'py>,
    wg_sources: Vec<usize>,
    wg_targets: Vec<usize>,
    wg_losses: Vec<f64>,
    laser_power_dbm: f64,
    detector_sensitivity_dbm: f64,
) -> PyResult<Py<PyAny>> {
    let wgs: Vec<(usize, usize, f64)> = wg_sources
        .into_iter()
        .zip(wg_targets)
        .zip(wg_losses)
        .map(|((s, t), l)| (s, t, l))
        .collect();

    let result =
        photonic::analyze_power_budget(&wgs, &[], laser_power_dbm, detector_sensitivity_dbm);

    let dict = PyDict::new(py);
    let margins: Vec<f64> = result.iter().map(|r| r.margin_db).collect();
    let passed: Vec<bool> = result.iter().map(|r| r.passed).collect();
    let total_losses: Vec<f64> = result.iter().map(|r| r.total_loss_db).collect();

    dict.set_item("margins_db", margins)?;
    dict.set_item("passed", passed)?;
    dict.set_item("total_losses_db", total_losses)?;
    dict.set_item("n_paths", result.len())?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Geometric crosstalk analysis for a uniform bank of parallel waveguides.
#[pyfunction]
#[pyo3(signature = (num_waveguides, gap_nm, coupling_length_um, wavelength_nm=1550.0, core_index=3.48, cladding_index=1.45))]
fn py_ph_analyze_crosstalk_bank<'py>(
    py: Python<'py>,
    num_waveguides: usize,
    gap_nm: f64,
    coupling_length_um: f64,
    wavelength_nm: f64,
    core_index: f64,
    cladding_index: f64,
) -> PyResult<Py<PyAny>> {
    let r = photonic::analyze_crosstalk_bank(
        num_waveguides,
        gap_nm,
        coupling_length_um,
        wavelength_nm,
        core_index,
        cladding_index,
    );
    let dict = PyDict::new(py);
    dict.set_item("num_waveguides", r.num_waveguides)?;
    dict.set_item("num_pairs", r.num_near_pairs + r.num_far_pairs)?;
    dict.set_item("num_near_pairs", r.num_near_pairs)?;
    dict.set_item("num_far_pairs", r.num_far_pairs)?;
    dict.set_item("gap_nm", r.gap_nm)?;
    dict.set_item("coupling_length_um", r.coupling_length_um)?;
    dict.set_item("adjacent_coupling_ratio", r.adjacent_coupling_ratio)?;
    dict.set_item("adjacent_isolation_db", r.adjacent_isolation_db)?;
    dict.set_item("next_nearest_coupling_ratio", r.next_nearest_coupling_ratio)?;
    dict.set_item("next_nearest_isolation_db", r.next_nearest_isolation_db)?;
    dict.set_item("worst_isolation_db", r.worst_isolation_db)?;
    dict.set_item("mean_coupling_ratio", r.mean_coupling_ratio)?;
    dict.set_item("max_coupling_ratio", r.max_coupling_ratio)?;
    dict.set_item("crosstalk_safe", r.crosstalk_safe)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

/// Per-pair geometric crosstalk for arbitrary waveguide geometry.
/// `pairs_a[i]`, `pairs_b[i]`, `gaps_nm[i]`, `lengths_um[i]` describe pair i.
/// Evaluated in parallel via Rayon — the O(N²) analysis path.
#[pyfunction]
#[pyo3(signature = (pairs_a, pairs_b, gaps_nm, lengths_um, wavelength_nm=1550.0, core_index=3.48, cladding_index=1.45))]
fn py_ph_analyze_crosstalk_pairs<'py>(
    py: Python<'py>,
    pairs_a: Vec<usize>,
    pairs_b: Vec<usize>,
    gaps_nm: Vec<f64>,
    lengths_um: Vec<f64>,
    wavelength_nm: f64,
    core_index: f64,
    cladding_index: f64,
) -> PyResult<Py<PyAny>> {
    let n = pairs_a.len();
    if pairs_b.len() != n || gaps_nm.len() != n || lengths_um.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "pairs_a, pairs_b, gaps_nm, lengths_um must be equal length",
        ));
    }
    let pairs: Vec<(usize, usize, f64, f64)> = pairs_a
        .into_iter()
        .zip(pairs_b)
        .zip(gaps_nm)
        .zip(lengths_um)
        .map(|(((a, b), g), l)| (a, b, g, l))
        .collect();
    let results =
        photonic::analyze_crosstalk_pairs(&pairs, wavelength_nm, core_index, cladding_index);

    let dict = PyDict::new(py);
    let idx_a: Vec<usize> = results.iter().map(|r| r.index_a).collect();
    let idx_b: Vec<usize> = results.iter().map(|r| r.index_b).collect();
    let gaps: Vec<f64> = results.iter().map(|r| r.gap_nm).collect();
    let lens: Vec<f64> = results.iter().map(|r| r.coupling_length_um).collect();
    let kappas: Vec<f64> = results
        .iter()
        .map(|r| r.coupling_coefficient_per_um)
        .collect();
    let ratios: Vec<f64> = results.iter().map(|r| r.coupling_ratio).collect();
    let isos: Vec<f64> = results.iter().map(|r| r.isolation_db).collect();

    dict.set_item("pair_a", idx_a)?;
    dict.set_item("pair_b", idx_b)?;
    dict.set_item("gap_nm", gaps)?;
    dict.set_item("coupling_length_um", lens)?;
    dict.set_item("coupling_coefficient_per_um", kappas)?;
    dict.set_item("coupling_ratio", ratios)?;
    dict.set_item("isolation_db", isos)?;
    dict.set_item("num_pairs", n)?;
    dict.set_item("backend", "rust")?;
    Ok(dict.into_any().unbind())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_ph_route_waveguides, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_mzi_transfer_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_cascade_mzi, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_analyze_crosstalk, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_analyze_power_budget, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_analyze_crosstalk_bank, m)?)?;
    m.add_function(wrap_pyfunction!(py_ph_analyze_crosstalk_pairs, m)?)?;
    Ok(())
}
