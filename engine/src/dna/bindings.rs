// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — DNA acceleration PyO3 bindings

//! Python bindings for the DNA strand-displacement acceleration primitives.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
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

/// Check real strand pairs only after admitting workers; fewer than two stays lazy.
#[pyfunction]
fn py_dna_check_cross_hybridization<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    threshold: usize,
) -> PyResult<Py<PyAny>> {
    if sequences.len() >= 2 {
        crate::runtime_control_binding::ensure_parallel_runtime()?;
    }
    let seqs: Vec<Vec<u8>> = sequences.into_iter().map(|s| s.into_bytes()).collect();
    let flags = dna::check_cross_hybridization(&seqs, threshold);
    let result: Vec<Py<PyAny>> = flags
        .into_iter()
        .map(|(i, j, score)| {
            let d = PyDict::new(py);
            d.set_item("strand_a", i)?;
            d.set_item("strand_b", j)?;
            d.set_item("score", score)?;
            Ok(d.into_any().unbind())
        })
        .collect::<PyResult<_>>()?;
    Ok(result.into_pyobject(py)?.into())
}

/// Simulate a DNA gate circuit's kinetics; the circuit is given as lists with one entry per gate.
///
/// Gate types are read without regard to case. Returns a dict of NumPy arrays: ``"time"``, sample
/// ``t`` at ``t * dt``, and one trace per gate output, in nM.
///
/// Raises ``ValueError`` before anything is computed when the gate lists differ in length,
/// ``input_names`` and ``input_concs`` differ in length, a name is given twice, a gate type is
/// unknown, or the core refuses the run: ``dt`` is not finite and above 0, ``duration_s`` is not
/// finite and at least ``dt``, a concentration is not finite or is below 0, ``NAND`` and ``XOR``
/// have no rate law, a gate has an input name count its kind does not read, an output is
/// ``"time"`` or another gate's, or the run would hold more than 2**25 samples.
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
    let lengths = [
        gate_types.len(),
        gate_inputs.len(),
        gate_outputs.len(),
        gate_thresholds.len(),
        gate_leaks.len(),
    ];
    if lengths.iter().any(|&length| length != gate_types.len()) {
        return Err(PyValueError::new_err(format!(
            "gate_types, gate_inputs, gate_outputs, gate_thresholds and gate_leaks must have one \
             entry per gate; got {}, {}, {}, {}, {}",
            lengths[0], lengths[1], lengths[2], lengths[3], lengths[4]
        )));
    }
    if input_names.len() != input_concs.len() {
        return Err(PyValueError::new_err(format!(
            "input_names and input_concs must have the same length; got {} and {}",
            input_names.len(),
            input_concs.len()
        )));
    }

    let mut inputs = std::collections::HashMap::with_capacity(input_names.len());
    for (name, conc) in input_names.into_iter().zip(input_concs) {
        if inputs.contains_key(&name) {
            return Err(PyValueError::new_err(format!(
                "input name '{name}' is given twice"
            )));
        }
        inputs.insert(name, conc);
    }

    let mut gates = Vec::with_capacity(gate_types.len());
    let specs = gate_types
        .iter()
        .zip(gate_inputs)
        .zip(gate_outputs)
        .zip(gate_thresholds)
        .zip(gate_leaks);
    for (index, ((((kind, input_names), output_name), threshold), leak_rate)) in specs.enumerate() {
        let gate_type = dna::DnaGateType::parse(kind).ok_or_else(|| {
            PyValueError::new_err(format!("gate {index} has the unknown type '{kind}'"))
        })?;
        gates.push(dna::DnaGateSpec {
            gate_type,
            input_names,
            output_name,
            threshold,
            leak_rate,
        });
    }

    let config = dna::KineticConfig {
        k_hyb,
        k_disp,
        temperature_c,
        max_conc: 200.0,
        use_rk4,
    };

    let result = dna::simulate_kinetics(&gates, &inputs, duration_s, dt, &config)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

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
