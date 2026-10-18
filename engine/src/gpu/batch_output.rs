// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Python GPU batch output admission

//! Reject unrepresentable shader indices and device output storage before dispatch.

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::PyResult;

/// Admit nonempty four-byte batch outputs within host and requested device limits.
pub(super) fn validate(n_neurons: usize, n_steps: usize, limits: wgpu::Limits) -> PyResult<()> {
    if n_neurons == 0 || n_steps == 0 {
        return Ok(());
    }
    let count = n_neurons
        .checked_mul(n_steps)
        .ok_or_else(|| PyMemoryError::new_err("GPU batch output cannot be allocated"))?;
    let count = u32::try_from(count)
        .map_err(|_| PyValueError::new_err("GPU batch dimensions exceed u32 shader indexing"))?;
    let max_bytes = limits
        .max_buffer_size
        .min(u64::from(limits.max_storage_buffer_binding_size))
        .min(isize::MAX as u64);
    if u64::from(count) * 4 > max_bytes {
        return Err(PyMemoryError::new_err(
            "GPU batch output cannot be allocated",
        ));
    }
    // Both batch kernels dispatch one 64-thread workgroup per neuron block.
    if n_neurons.div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize {
        return Err(PyValueError::new_err(
            "GPU batch dispatch exceeds device workgroup limit",
        ));
    }
    Ok(())
}
