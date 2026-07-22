// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Direction-selective retinal-ganglion-cell PyO3 binding

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[pyclass(
    name = "RustDirectionSelectiveRGC",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyDirectionSelectiveRGC {
    inner: neurons::DirectionSelectiveRGC,
}

#[pymethods]
impl PyDirectionSelectiveRGC {
    #[new]
    #[pyo3(signature = (is_on=true))]
    fn new(is_on: bool) -> Self {
        Self {
            inner: if is_on {
                neurons::DirectionSelectiveRGC::new_on()
            } else {
                neurons::DirectionSelectiveRGC::new_off()
            },
        }
    }

    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }

    fn step_rf(&mut self, intensity: f64, surround_mean: f64) -> i32 {
        self.inner.step_rf(intensity, surround_mean)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("is_on_centre", self.inner.is_on_centre)?;
        Ok(d.into_any().unbind())
    }
}

/// Register the direction-selective retinal-ganglion-cell class.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDirectionSelectiveRGC>()?;
    Ok(())
}
