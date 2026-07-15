// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DNA acceleration PyO3 bindings

//! Python bindings for the DNA strand-displacement acceleration primitives.

use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObject;

use crate::dna;

// ── DNA acceleration PyO3 wrappers ───────────────────────────────────

#[pyfunction]
fn py_dna_design_sequence(_py: Python<'_>, length: usize, seed: u64) -> String {
    String::from_utf8(dna::design_sequence(length, seed)).unwrap_or_default()
}

#[pyfunction]
fn py_dna_design_orthogonal_set(
    _py: Python<'_>,
    count: usize,
    length: usize,
    seed: u64,
) -> Vec<String> {
    dna::design_orthogonal_set(count, length, seed)
        .into_iter()
        .map(|s| String::from_utf8(s).unwrap_or_default())
        .collect()
}

#[pyfunction]
fn py_dna_check_cross_hybridization<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    threshold: usize,
) -> PyResult<Py<PyAny>> {
    let seqs: Vec<Vec<u8>> = sequences.into_iter().map(|s| s.into_bytes()).collect();
    let flags = dna::check_cross_hybridization(&seqs, threshold);
    let result: Vec<Py<PyAny>> = flags
        .into_iter()
        .map(|(i, j, score)| {
            let d = PyDict::new(py);
            d.set_item("strand_a", i).unwrap();
            d.set_item("strand_b", j).unwrap();
            d.set_item("score", score).unwrap();
            d.into_any().unbind()
        })
        .collect();
    Ok(result.into_pyobject(py)?.into())
}

#[pyfunction]
#[pyo3(signature = (
    gate_types, gate_inputs, gate_outputs, gate_thresholds, gate_leaks,
    input_names, input_concs, duration_s=1800.0, dt=1.0,
    k_hyb=3e5, k_disp=1.0, temperature_c=37.0, use_rk4=true
))]
#[allow(clippy::too_many_arguments)]
fn py_dna_simulate_kinetics<'py>(
    py: Python<'py>,
    gate_types: Vec<String>,
    gate_inputs: Vec<Vec<String>>,
    gate_outputs: Vec<String>,
    gate_thresholds: Vec<f64>,
    gate_leaks: Vec<f64>,
    input_names: Vec<String>,
    input_concs: Vec<f64>,
    duration_s: f64,
    dt: f64,
    k_hyb: f64,
    k_disp: f64,
    temperature_c: f64,
    use_rk4: bool,
) -> PyResult<Py<PyAny>> {
    let gates: Vec<dna::DnaGateSpec> = gate_types
        .iter()
        .zip(gate_inputs.iter())
        .zip(gate_outputs.iter())
        .zip(gate_thresholds.iter())
        .zip(gate_leaks.iter())
        .map(|((((gt, gi), go), th), lk)| {
            let gate_type = match gt.to_uppercase().as_str() {
                "AND" => dna::DnaGateType::And,
                "OR" => dna::DnaGateType::Or,
                "NOT" => dna::DnaGateType::Not,
                "THRESHOLD" => dna::DnaGateType::Threshold,
                "MUX" => dna::DnaGateType::Mux,
                "AMPLIFIER" => dna::DnaGateType::Amplifier,
                "BUFFER" => dna::DnaGateType::Buffer,
                "NAND" => dna::DnaGateType::Nand,
                "XOR" => dna::DnaGateType::Xor,
                _ => dna::DnaGateType::And,
            };
            dna::DnaGateSpec {
                gate_type,
                input_names: gi.clone(),
                output_name: go.clone(),
                threshold: *th,
                leak_rate: *lk,
            }
        })
        .collect();

    let mut inputs = std::collections::HashMap::new();
    for (name, conc) in input_names.into_iter().zip(input_concs) {
        inputs.insert(name, conc);
    }

    let config = dna::KineticConfig {
        k_hyb,
        k_disp,
        temperature_c,
        max_conc: 200.0,
        use_rk4,
    };

    let result = dna::simulate_kinetics(&gates, &inputs, duration_s, dt, &config);

    let dict = PyDict::new(py);
    for (key, trace) in result {
        dict.set_item(key, trace.into_pyarray(py))?;
    }
    Ok(dict.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (sequence, min_stem=4, min_loop=3))]
fn py_dna_detect_hairpins(
    _py: Python<'_>,
    sequence: &str,
    min_stem: usize,
    min_loop: usize,
) -> Vec<(usize, usize, usize)> {
    dna::detect_hairpins(sequence.as_bytes(), min_stem, min_loop)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_dna_design_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(py_dna_design_orthogonal_set, m)?)?;
    m.add_function(wrap_pyfunction!(py_dna_check_cross_hybridization, m)?)?;
    m.add_function(wrap_pyfunction!(py_dna_simulate_kinetics, m)?)?;
    m.add_function(wrap_pyfunction!(py_dna_detect_hairpins, m)?)?;
    Ok(())
}
