// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Engine runtime-control PyO3 bindings

//! Python bindings for runtime SIMD discovery and Rayon thread-pool control.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Register engine runtime-control functions with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(simd_tier, module)?)?;
    module.add_function(wrap_pyfunction!(set_num_threads, module)?)?;
    Ok(())
}

/// Returns the highest SIMD tier available on this CPU.
#[pyfunction]
fn simd_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            return "avx512-vpopcntdq";
        }
        if is_x86_feature_detected!("avx512bw") {
            return "avx512bw";
        }
        if is_x86_feature_detected!("avx512f") {
            return "avx512f";
        }
        if is_x86_feature_detected!("avx2") {
            return "avx2";
        }
        if is_x86_feature_detected!("popcnt") {
            return "popcnt";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    "portable"
}

/// Set the number of threads in the global rayon thread pool.
///
/// Must be called before any parallel operation.
/// Passing 0 uses rayon's default (number of CPU cores).
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    if n == 0 {
        return Ok(());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| PyValueError::new_err(format!("Cannot set thread pool: {e}")))
}
