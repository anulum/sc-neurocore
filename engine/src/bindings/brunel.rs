// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Brunel-network PyO3 binding

//! Python binding for the fixed-point Brunel network simulator.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::brunel;
use crate::fixed_point_lif_binding::array_allocation::zeros_1d;

/// Register the fixed-point Brunel network with the extension module.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBrunelNetwork>()?;
    Ok(())
}

#[pyclass(
    name = "FixedPointBrunelNetwork",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
/// Fixed-point CSR network with owned connectivity and seeded Poisson drive.
pub struct PyBrunelNetwork {
    inner: brunel::BrunelNetwork,
}

#[pymethods]
impl PyBrunelNetwork {
    #[new]
    #[pyo3(signature = (
        n_neurons,
        w_indptr,
        w_indices,
        w_data,
        leak_k,
        gain_k,
        ext_lambda,
        ext_weight_fp,
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2,
        seed=42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_neurons: usize,
        w_indptr: PyReadonlyArray1<'_, i64>,
        w_indices: PyReadonlyArray1<'_, i64>,
        w_data: PyReadonlyArray1<'_, i16>,
        leak_k: i16,
        gain_k: i16,
        ext_lambda: f64,
        ext_weight_fp: i16,
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        seed: u64,
    ) -> PyResult<Self> {
        let indptr = w_indptr
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_indptr: {e}")))?;
        let indices = w_indices
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_indices: {e}")))?;
        let data = w_data
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("Cannot read w_data: {e}")))?;

        let mut row_offsets = reserved_buffer(indptr.len())?;
        for &value in indptr {
            row_offsets.push(
                usize::try_from(value)
                    .map_err(|_| PyValueError::new_err("w_indptr values must be nonnegative"))?,
            );
        }
        let mut col_indices = reserved_buffer(indices.len())?;
        for &value in indices {
            col_indices.push(
                usize::try_from(value)
                    .map_err(|_| PyValueError::new_err("w_indices values must be nonnegative"))?,
            );
        }
        let mut values = reserved_buffer(data.len())?;
        values.extend_from_slice(data);

        let inner = brunel::BrunelNetwork::new(
            n_neurons,
            row_offsets,
            col_indices,
            values,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
            leak_k,
            gain_k,
            ext_lambda,
            ext_weight_fp,
            seed,
        )
        .map_err(|error| {
            if error == brunel::ALLOCATION_ERROR {
                PyMemoryError::new_err(error)
            } else {
                PyValueError::new_err(error)
            }
        })?;

        Ok(Self { inner })
    }

    /// Allocate all output and working storage before advancing the seeded network.
    fn run<'py>(&mut self, py: Python<'py>, n_steps: usize) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if n_steps > isize::MAX as usize / std::mem::size_of::<u32>() {
            return Err(PyMemoryError::new_err(brunel::ALLOCATION_ERROR));
        }
        let output = zeros_1d::<u32>(py, n_steps).map_err(|error| {
            if error.is_instance_of::<PyMemoryError>(py) {
                PyMemoryError::new_err(brunel::ALLOCATION_ERROR)
            } else {
                error
            }
        })?;
        let mut borrowed = output.try_readwrite()?;
        self.inner
            .try_run_into(borrowed.as_slice_mut()?)
            .map_err(|_| PyMemoryError::new_err(brunel::ALLOCATION_ERROR))?;
        drop(borrowed);
        Ok(output)
    }
}

/// Reserve an owned CSR copy before copying borrowed Python values.
fn reserved_buffer<T>(length: usize) -> PyResult<Vec<T>> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(length)
        .map_err(|_| PyMemoryError::new_err(brunel::ALLOCATION_ERROR))?;
    Ok(buffer)
}
