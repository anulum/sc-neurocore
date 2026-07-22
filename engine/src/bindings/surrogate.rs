// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Surrogate-gradient LIF PyO3 binding

//! Python binding for surrogate-gradient LIF training and shared surrogate parsing.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::grad;

/// Register surrogate-gradient LIF training with the extension module.
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySurrogateLif>()?;
    Ok(())
}

pub(crate) fn parse_surrogate(name: &str, k: Option<f32>) -> PyResult<grad::SurrogateType> {
    let normalized = name.to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        "fast_sigmoid" => Ok(grad::SurrogateType::FastSigmoid {
            k: k.unwrap_or(25.0),
        }),
        "superspike" | "super_spike" => Ok(grad::SurrogateType::SuperSpike {
            k: k.unwrap_or(100.0),
        }),
        "arctan" | "arc_tan" => Ok(grad::SurrogateType::ArcTan { k: k.unwrap_or(10.0) }),
        "straightthrough" | "straight_through" | "ste" => Ok(grad::SurrogateType::StraightThrough),
        _ => Err(PyValueError::new_err(format!(
            "Unknown surrogate '{}'. Use one of: fast_sigmoid, superspike, arctan, straight_through.",
            name
        ))),
    }
}

#[pyclass(
    name = "SurrogateLif",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
pub struct PySurrogateLif {
    inner: grad::SurrogateLif,
}

#[pymethods]
impl PySurrogateLif {
    #[new]
    #[pyo3(signature = (
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2,
        surrogate="fast_sigmoid",
        k=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
        surrogate: &str,
        k: Option<f32>,
    ) -> PyResult<Self> {
        let surrogate = parse_surrogate(surrogate, k)?;
        Ok(Self {
            inner: grad::SurrogateLif::new(
                data_width,
                fraction,
                v_rest,
                v_reset,
                v_threshold,
                refractory_period,
                surrogate,
            ),
        })
    }

    #[pyo3(signature = (leak_k, gain_k, i_t, noise_in=0))]
    fn forward(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        self.inner.forward(leak_k, gain_k, i_t, noise_in)
    }

    fn backward(&mut self, grad_output: f32) -> PyResult<f32> {
        self.inner
            .backward(grad_output)
            .map_err(PyValueError::new_err)
    }

    fn clear_trace(&mut self) {
        self.inner.clear_trace();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn trace_len(&self) -> usize {
        self.inner.trace_len()
    }
}
