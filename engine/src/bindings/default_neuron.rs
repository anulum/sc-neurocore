// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Default-constructor neuron PyO3 class contract

/// Define the common PyO3 class surface for a default-constructible neuron.
macro_rules! py_neuron_default {
    ($pylit:literal, $pyname:ident, $rust:ty, fallible $method:ident, map $mapper:path $(, state $sname:ident)*) => {
        /// Python class for one default-constructed neuron with typed step failures.
        #[pyclass(name = $pylit, module = "sc_neurocore_engine.sc_neurocore_engine")]
        #[derive(Clone)]
        pub struct $pyname {
            inner: $rust,
        }

        #[pymethods]
        impl $pyname {
            /// Construct the catalogue defaults without accepting configuration arguments.
            #[new]
            fn new() -> Self {
                Self {
                    inner: <$rust>::default(),
                }
            }

            /// Advance the configured native step or preserve its typed refusal.
            fn step(&mut self, current: f64) -> PyResult<i32> {
                self.inner.$method(current).map_err($mapper)
            }

            /// Restore the core model's documented reset state.
            fn reset(&mut self) {
                self.inner.reset();
            }

            /// Return detached declared state fields without exposing mutable native storage.
            fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let d = PyDict::new(py);
                $(d.set_item(stringify!($sname), self.inner.$sname)?;)*
                Ok(d.into_any().unbind())
            }
        }
    };
    ($pylit:literal, $pyname:ident, $rust:ty $(, state $sname:ident)*) => {
        /// Python class for one default-constructed neuron model.
        #[pyclass(name = $pylit, module = "sc_neurocore_engine.sc_neurocore_engine")]
        #[derive(Clone)]
        pub struct $pyname {
            inner: $rust,
        }

        #[pymethods]
        impl $pyname {
            #[new]
            fn new() -> Self {
                Self {
                    inner: <$rust>::default(),
                }
            }

            fn step(&mut self, current: f64) -> i32 {
                self.inner.step(current)
            }

            fn reset(&mut self) {
                self.inner.reset();
            }

            fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let d = PyDict::new(py);
                $(d.set_item(stringify!($sname), self.inner.$sname)?;)*
                Ok(d.into_any().unbind())
            }
        }
    };
}
