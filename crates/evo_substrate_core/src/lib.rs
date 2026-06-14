// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Evolutionary Substrate Core (Rust FFI)
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! # Evolutionary Substrate Core
//!
//! Hot-path kernels for `src/sc_neurocore/evo_substrate/evo_substrate.py`:
//! - `genomic_distance` — scale-invariant L1 on 19-D genome vectors
//! - `crossover_uniform` — Syswerda uniform crossover
//! - `point_mutation` — Gaussian multiplicative point mutation
//! - `population_diversity` — mean pairwise distance
//!
//! The Python module remains the reference implementation (ships even
//! when the compiled `.so` is missing). Numerical parity is bit-exact
//! on the distance and crossover kernels; point mutation matches in
//! distribution (same Gaussian parameters) because it needs an RNG.
//!
//! The `runner` submodule exposes a *whole-process* industrial evolve
//! runner (`evolve_run`) that ports `ReplicationEngine.evolve_generation`
//! plus the 11 industrial guards (FormalSafetyGuard, BloatPenalizer,
//! AgeRegulator, ExtinctionDetector, HallOfFame, ParetoFront,
//! TournamentSelector, LineageTracker, MutationEngine × 4 variants,
//! CrossoverEngine, parametric FitnessEvaluator) to Rust so Python can
//! invoke one call that runs N generations of M organisms natively.

pub mod runner;

// ---------------------------------------------------------------------------
// Genomic distance — scale-invariant L1 on 19-D vectors
// ---------------------------------------------------------------------------

const EPSILON: f64 = 1e-10;

/// Scale-invariant L1 distance
/// `(1/D) Σ |aᵢ−bᵢ| / (|aᵢ|+|bᵢ|+ε)`.
///
/// Matches `sc_neurocore.evo_substrate.evo_substrate.genomic_distance`.
#[inline]
pub fn genomic_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for (ai, bi) in a.iter().zip(b.iter()) {
        let diff = (ai - bi).abs();
        let norm = ai.abs() + bi.abs() + EPSILON;
        acc += diff / norm;
    }
    acc / (a.len() as f64)
}

/// Uniform crossover: for each coordinate, pick `a[i]` if `mask[i]==1`,
/// else `b[i]`. `mask` must have the same length as `a` and `b`.
#[inline]
pub fn crossover_uniform(a: &[f64], b: &[f64], mask: &[u8], out: &mut [f64]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(out.len(), a.len());
    debug_assert_eq!(mask.len(), a.len());
    for i in 0..a.len() {
        out[i] = if mask[i] != 0 { a[i] } else { b[i] };
    }
}

/// Gaussian multiplicative point mutation.
///
/// For each coordinate: if `mutation_mask[i]==1`, add
/// `noise[i] * (|gene[i]| + ε)` to `gene[i]`. `noise` should come from a
/// caller-provided Gaussian PRNG so the Rust kernel stays pure /
/// deterministic under a fixed seed on the caller's side.
#[inline]
pub fn point_mutation(gene: &mut [f64], mutation_mask: &[u8], noise: &[f64]) {
    debug_assert_eq!(gene.len(), mutation_mask.len());
    debug_assert_eq!(gene.len(), noise.len());
    for i in 0..gene.len() {
        if mutation_mask[i] != 0 {
            gene[i] += noise[i] * (gene[i].abs() + 1e-8);
        }
    }
}

/// Mean pairwise distance over a population.
/// `population` is a flat `n × d` row-major matrix.
#[inline]
pub fn population_diversity(population: &[f64], n: usize, d: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut count = 0.0;
    for i in 0..n {
        let row_i = &population[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let row_j = &population[j * d..(j + 1) * d];
            acc += genomic_distance(row_i, row_j);
            count += 1.0;
        }
    }
    acc / count
}

// ---------------------------------------------------------------------------
// C-FFI exports
// ---------------------------------------------------------------------------

#[no_mangle]
/// # Safety
/// `a_ptr` and `b_ptr` must be valid for `len` f64 elements each.
pub unsafe extern "C" fn genomic_distance_ffi(
    a_ptr: *const f64,
    b_ptr: *const f64,
    len: usize,
) -> f64 {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, len) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, len) };
    genomic_distance(a, b)
}

#[no_mangle]
/// # Safety
/// All three pointers must be valid for `len` elements of their element types.
/// `out_ptr` must be writable for `len` f64 elements.
pub unsafe extern "C" fn crossover_uniform_ffi(
    a_ptr: *const f64,
    b_ptr: *const f64,
    mask_ptr: *const u8,
    out_ptr: *mut f64,
    len: usize,
) {
    let a = unsafe { std::slice::from_raw_parts(a_ptr, len) };
    let b = unsafe { std::slice::from_raw_parts(b_ptr, len) };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, len) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
    crossover_uniform(a, b, mask, out);
}

#[no_mangle]
/// # Safety
/// `gene_ptr` must be writable for `len` f64 elements; `mask_ptr` valid
/// for `len` u8 elements; `noise_ptr` valid for `len` f64 elements.
pub unsafe extern "C" fn point_mutation_ffi(
    gene_ptr: *mut f64,
    mask_ptr: *const u8,
    noise_ptr: *const f64,
    len: usize,
) {
    let gene = unsafe { std::slice::from_raw_parts_mut(gene_ptr, len) };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, len) };
    let noise = unsafe { std::slice::from_raw_parts(noise_ptr, len) };
    point_mutation(gene, mask, noise);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_identical_is_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(genomic_distance(&a, &a), 0.0);
    }

    #[test]
    fn distance_symmetry() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let d_ab = genomic_distance(&a, &b);
        let d_ba = genomic_distance(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-12);
    }

    #[test]
    fn distance_scale_invariance() {
        // |a-b|/(|a|+|b|) is invariant under uniform scaling in sign-matched
        // coordinates.
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0];
        let aa: Vec<f64> = a.iter().map(|x| x * 100.0).collect();
        let bb: Vec<f64> = b.iter().map(|x| x * 100.0).collect();
        let d1 = genomic_distance(&a, &b);
        let d2 = genomic_distance(&aa, &bb);
        assert!((d1 - d2).abs() < 1e-8);
    }

    #[test]
    fn crossover_respects_mask() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![-1.0, -2.0, -3.0, -4.0];
        let mask = vec![1u8, 0, 1, 0];
        let mut out = vec![0.0; 4];
        crossover_uniform(&a, &b, &mask, &mut out);
        assert_eq!(out, vec![1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn point_mutation_zero_mask_is_identity() {
        let mut gene = vec![1.0, 2.0, 3.0];
        let mask = vec![0u8, 0, 0];
        let noise = vec![10.0, 10.0, 10.0];
        let before = gene.clone();
        point_mutation(&mut gene, &mask, &noise);
        assert_eq!(gene, before);
    }

    #[test]
    fn point_mutation_applies_multiplicative_noise() {
        let mut gene = vec![1.0, 2.0];
        let mask = vec![1u8, 1];
        let noise = vec![0.1, 0.1];
        point_mutation(&mut gene, &mask, &noise);
        let expected_0 = 1.0 + 0.1 * (1.0 + 1e-8);
        let expected_1 = 2.0 + 0.1 * (2.0 + 1e-8);
        assert!((gene[0] - expected_0).abs() < 1e-9);
        assert!((gene[1] - expected_1).abs() < 1e-9);
    }

    #[test]
    fn diversity_single_genome_is_zero() {
        let pop = vec![1.0, 2.0, 3.0];
        assert_eq!(population_diversity(&pop, 1, 3), 0.0);
    }

    #[test]
    fn diversity_two_genomes_equals_distance() {
        let pop = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = genomic_distance(&pop[..3], &pop[3..]);
        let div = population_diversity(&pop, 2, 3);
        assert!((d - div).abs() < 1e-12);
    }

    #[test]
    fn ffi_distance_matches_safe_fn() {
        let a = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let b = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let safe = genomic_distance(&a, &b);
        let ffi = unsafe { genomic_distance_ffi(a.as_ptr(), b.as_ptr(), a.len()) };
        assert!((safe - ffi).abs() < 1e-14);
    }
}

// ---------------------------------------------------------------------------
// PyO3 bindings (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "pyo3_bindings")]
mod python {
    use super::*;
    use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::prelude::*;

    /// Scale-invariant L1 distance between two 1-D float arrays.
    #[pyfunction]
    fn py_genomic_distance<'py>(
        a: PyReadonlyArray1<'py, f64>,
        b: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<f64> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        if a_slice.len() != b_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "genome length mismatch",
            ));
        }
        Ok(genomic_distance(a_slice, b_slice))
    }

    /// Uniform crossover. Returns a fresh NumPy array.
    #[pyfunction]
    fn py_crossover_uniform<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<'py, f64>,
        b: PyReadonlyArray1<'py, f64>,
        mask: PyReadonlyArray1<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let a_slice = a.as_slice()?;
        let b_slice = b.as_slice()?;
        let mask_slice = mask.as_slice()?;
        if a_slice.len() != b_slice.len() || a_slice.len() != mask_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "length mismatch among a, b, mask",
            ));
        }
        let mut out = vec![0.0f64; a_slice.len()];
        crossover_uniform(a_slice, b_slice, mask_slice, &mut out);
        Ok(PyArray1::from_vec(py, out))
    }

    /// In-place Gaussian multiplicative point mutation. Caller supplies the
    /// noise vector (sampled from its own PRNG) so the Rust path stays
    /// pure and deterministic.
    #[pyfunction]
    fn py_point_mutation<'py>(
        py: Python<'py>,
        gene: PyReadonlyArray1<'py, f64>,
        mask: PyReadonlyArray1<'py, u8>,
        noise: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let gene_slice = gene.as_slice()?;
        let mask_slice = mask.as_slice()?;
        let noise_slice = noise.as_slice()?;
        if gene_slice.len() != mask_slice.len() || gene_slice.len() != noise_slice.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "length mismatch among gene, mask, noise",
            ));
        }
        let mut out = gene_slice.to_vec();
        point_mutation(&mut out, mask_slice, noise_slice);
        Ok(PyArray1::from_vec(py, out))
    }

    /// Mean pairwise distance over a population matrix (`n × d`).
    #[pyfunction]
    fn py_population_diversity<'py>(population: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        let shape = population.shape();
        let n = shape[0];
        let d = shape[1];
        let flat = population.as_slice()?;
        Ok(population_diversity(flat, n, d))
    }

    /// Run the full industrial evolve loop natively in Rust. The config
    /// arrives as a JSON string (matching `runner::EvolveConfig`) and the
    /// result is returned as a JSON string (matching `runner::EvolveResult`).
    /// JSON is chosen to keep the FFI surface stable across pyo3 versions
    /// and across the Julia / Go / Mojo runners that will share the same
    /// wire format.
    #[pyfunction]
    fn py_evolve_run(config_json: &str) -> PyResult<String> {
        let cfg: crate::runner::EvolveConfig = serde_json::from_str(config_json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid EvolveConfig JSON: {e}"))
        })?;
        let result = crate::runner::evolve_run(&cfg);
        serde_json::to_string(&result).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("EvolveResult serialise failed: {e}"))
        })
    }

    /// Python module registration.
    #[pymodule]
    fn evo_substrate_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_genomic_distance, m)?)?;
        m.add_function(wrap_pyfunction!(py_crossover_uniform, m)?)?;
        m.add_function(wrap_pyfunction!(py_point_mutation, m)?)?;
        m.add_function(wrap_pyfunction!(py_population_diversity, m)?)?;
        m.add_function(wrap_pyfunction!(py_evolve_run, m)?)?;
        Ok(())
    }
}
