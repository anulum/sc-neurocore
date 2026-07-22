// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 wrappers for all neuron models

//! PyO3 wrappers and compatibility exports for all neuron models.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[macro_use]
#[path = "bindings/default_neuron.rs"]
mod default_neuron_binding;

#[path = "bindings/adex_neuron.rs"]
mod adex_neuron_binding;
#[path = "bindings/alpha.rs"]
mod alpha_binding;
#[path = "bindings/ermentrout_kopell_pop.rs"]
mod ermentrout_kopell_pop_binding;
#[path = "bindings/jansen_rit.rs"]
mod jansen_rit_binding;
#[path = "bindings/lapicque_neuron.rs"]
mod lapicque_neuron_binding;
#[path = "bindings/mcculloch_pitts.rs"]
mod mcculloch_pitts_binding;
#[path = "bindings/sigmoid_rate.rs"]
mod sigmoid_rate_binding;
#[path = "bindings/threshold_linear_rate.rs"]
mod threshold_linear_rate_binding;
#[path = "bindings/wong_wang.rs"]
mod wong_wang_binding;

#[path = "bindings/biophysical/mod.rs"]
mod biophysical_bindings;
#[path = "bindings/hardware/mod.rs"]
mod hardware_bindings;
#[path = "bindings/maps/mod.rs"]
mod map_bindings;
#[path = "bindings/multi_compartment/mod.rs"]
mod multi_compartment_bindings;
#[path = "bindings/simple_spiking/mod.rs"]
mod simple_spiking_bindings;
#[path = "bindings/stochastic/mod.rs"]
mod stochastic_bindings;
#[path = "bindings/trivial/mod.rs"]
mod trivial_bindings;

#[path = "bindings/adaptive_threshold_moe_neuron.rs"]
mod adaptive_threshold_moe_neuron_binding;
#[path = "bindings/arcane_neuron.rs"]
mod arcane_neuron_binding;
#[path = "bindings/attention_gated_neuron.rs"]
mod attention_gated_neuron_binding;
#[path = "bindings/cerebellar/mod.rs"]
mod cerebellar_bindings;
#[path = "bindings/channels/mod.rs"]
mod channel_bindings;
#[path = "bindings/compositional_binding_neuron.rs"]
mod compositional_binding_neuron_binding;
#[path = "bindings/continuous_attractor_neuron.rs"]
mod continuous_attractor_neuron_binding;
#[path = "bindings/differentiable_surrogate_neuron.rs"]
mod differentiable_surrogate_neuron_binding;
#[path = "bindings/hybrid_linear_attention_neuron.rs"]
mod hybrid_linear_attention_neuron_binding;
#[path = "bindings/interneurons/mod.rs"]
mod interneuron_bindings;
#[path = "bindings/meta_plastic_neuron.rs"]
mod meta_plastic_neuron_binding;
#[path = "bindings/misc/mod.rs"]
mod misc_bindings;
#[path = "bindings/motor/mod.rs"]
mod motor_bindings;
#[path = "bindings/multi_timescale_neuron.rs"]
mod multi_timescale_neuron_binding;
#[path = "bindings/population/mod.rs"]
mod population_bindings;
#[path = "bindings/predictive_coding_neuron.rs"]
mod predictive_coding_neuron_binding;
#[path = "bindings/quantum_inspired_lif_neuron.rs"]
mod quantum_inspired_lif_neuron_binding;
#[path = "bindings/self_referential_neuron.rs"]
mod self_referential_neuron_binding;
#[path = "bindings/sensory/mod.rs"]
mod sensory_bindings;
#[path = "bindings/synapses/mod.rs"]
mod synapse_bindings;

pub use biophysical_bindings::*;
pub use cerebellar_bindings::*;
pub use channel_bindings::*;
pub use hardware_bindings::*;
pub use interneuron_bindings::*;
pub use map_bindings::*;
pub use misc_bindings::*;
pub use motor_bindings::*;
pub use multi_compartment_bindings::*;
pub use population_bindings::*;
pub use sensory_bindings::*;
pub use simple_spiking_bindings::*;
pub use stochastic_bindings::*;
pub use synapse_bindings::*;
pub use trivial_bindings::*;

pub use adaptive_threshold_moe_neuron_binding::PyAdaptiveThresholdMoENeuron;
pub use adex_neuron_binding::PyAdExNeuron;
pub use alpha_binding::PyAlphaNeuron;
pub use arcane_neuron_binding::PyArcaneNeuron;
pub use attention_gated_neuron_binding::PyAttentionGatedNeuron;
pub use compositional_binding_neuron_binding::PyCompositionalBindingNeuron;
pub use continuous_attractor_neuron_binding::PyContinuousAttractorNeuron;
pub use differentiable_surrogate_neuron_binding::PyDifferentiableSurrogateNeuron;
pub use ermentrout_kopell_pop_binding::PyErmentroutKopellPopulation;
pub use hybrid_linear_attention_neuron_binding::PyHybridLinearAttentionNeuron;
pub use jansen_rit_binding::PyJansenRitUnit;
pub use lapicque_neuron_binding::PyLapicqueNeuron;
pub use mcculloch_pitts_binding::PyMcCullochPittsNeuron;
pub use meta_plastic_neuron_binding::PyMetaPlasticNeuron;
pub use multi_timescale_neuron_binding::PyMultiTimescaleNeuron;
pub use predictive_coding_neuron_binding::PyPredictiveCodingNeuron;
pub use quantum_inspired_lif_neuron_binding::PyQuantumInspiredLIFNeuron;
pub use self_referential_neuron_binding::PySelfReferentialNeuron;
pub use sigmoid_rate_binding::PySigmoidRateNeuron;
pub use threshold_linear_rate_binding::PyThresholdLinearRateNeuron;

#[pyclass(
    name = "EPropALIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyEPropALIFNeuron {
    inner: neurons::EPropALIFNeuron,
}

#[pymethods]
impl PyEPropALIFNeuron {
    #[new]
    #[pyo3(signature = (tau_m=20.0, tau_a=200.0, dt=1.0))]
    fn new(tau_m: f64, tau_a: f64, dt: f64) -> Self {
        Self {
            inner: neurons::EPropALIFNeuron::new(tau_m, tau_a, dt),
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
        d.set_item("v", self.inner.v)?;
        d.set_item("a", self.inner.a)?;
        d.set_item("e_trace", self.inner.e_trace)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "SuperSpikeNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySuperSpikeNeuron {
    inner: neurons::SuperSpikeNeuron,
}

#[pymethods]
impl PySuperSpikeNeuron {
    #[new]
    #[pyo3(signature = (tau_m=10.0, tau_e=10.0, dt=1.0))]
    fn new(tau_m: f64, tau_e: f64, dt: f64) -> Self {
        Self {
            inner: neurons::SuperSpikeNeuron::new(tau_m, tau_e, dt),
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
        d.set_item("v", self.inner.v)?;
        d.set_item("trace", self.inner.trace)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "BendaHerzNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyBendaHerzNeuron {
    inner: neurons::BendaHerzNeuron,
}

#[pymethods]
impl PyBendaHerzNeuron {
    #[new]
    #[pyo3(signature = (seed=42))]
    fn new(seed: u64) -> Self {
        Self {
            inner: neurons::BendaHerzNeuron::new(seed),
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
        d.set_item("a", self.inner.a)?;
        Ok(d.into_any().unbind())
    }
}

py_neuron_default!("BrunelWangNeuron", PyBrunelWangNeuron, neurons::BrunelWangNeuron, state v, state ref_remaining);

#[pyclass(
    name = "WilsonCowanUnit",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyWilsonCowanUnit {
    inner: neurons::WilsonCowanUnit,
}

#[pymethods]
impl PyWilsonCowanUnit {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::WilsonCowanUnit::new(),
        }
    }
    #[pyo3(signature = (ext_input=0.0))]
    fn step(&mut self, ext_input: f64) -> f64 {
        self.inner.step(ext_input)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("e", self.inner.e)?;
        d.set_item("i", self.inner.i)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "WongWangUnit",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyWongWangUnit {
    inner: neurons::WongWangUnit,
}

#[pymethods]
impl PyWongWangUnit {
    #[new]
    #[pyo3(signature = (seed=42))]
    fn new(seed: u64) -> Self {
        Self {
            inner: neurons::WongWangUnit::new(seed),
        }
    }
    #[pyo3(signature = (stim1=0.0, stim2=0.0))]
    fn step(&mut self, stim1: f64, stim2: f64) -> PyResult<(f64, f64)> {
        self.inner.step(stim1, stim2).map_err(PyValueError::new_err)
    }
    #[pyo3(signature = (stim1=0.0, stim2=0.0, xi1=0.0, xi2=0.0))]
    fn step_with_gaussian_samples(
        &mut self,
        stim1: f64,
        stim2: f64,
        xi1: f64,
        xi2: f64,
    ) -> PyResult<(f64, f64)> {
        self.inner
            .step_with_gaussian_samples(stim1, stim2, xi1, xi2)
            .map_err(PyValueError::new_err)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("s1", self.inner.s1)?;
        d.set_item("s2", self.inner.s2)?;
        d.set_item("noise1", self.inner.noise1)?;
        d.set_item("noise2", self.inner.noise2)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "WendlingNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyWendlingNeuron {
    inner: neurons::WendlingNeuron,
}

#[pymethods]
impl PyWendlingNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::WendlingNeuron::new(),
        }
    }
    #[pyo3(signature = (p_ext=220.0))]
    fn step(&mut self, p_ext: f64) -> f64 {
        self.inner.step(p_ext)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pyclass(
    name = "LarterBreakspearNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLarterBreakspearNeuron {
    inner: neurons::LarterBreakspearNeuron,
}

#[pymethods]
impl PyLarterBreakspearNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::LarterBreakspearNeuron::new(),
        }
    }
    #[pyo3(signature = (coupling=0.0))]
    fn step(&mut self, coupling: f64) -> f64 {
        self.inner.step(coupling)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("w", self.inner.w)?;
        d.set_item("z", self.inner.z)?;
        Ok(d.into_any().unbind())
    }
}

// ═══════════════════════════════════════════════════════════════════
// rate.rs models
// ═══════════════════════════════════════════════════════════════════

// AstrocyteModel: step returns f64
#[pyclass(
    name = "AstrocyteModel",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAstrocyteModel {
    inner: neurons::AstrocyteModel,
}

#[pymethods]
impl PyAstrocyteModel {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::AstrocyteModel::new(),
        }
    }
    fn step(&mut self, current: f64) -> f64 {
        self.inner.step(current)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("ca", self.inner.ca)?;
        d.set_item("h", self.inner.h)?;
        d.set_item("ip3", self.inner.ip3)?;
        Ok(d.into_any().unbind())
    }
}

// TsodyksMarkramNeuron: step(current, presynaptic_spike)
#[pyclass(
    name = "TsodyksMarkramNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTsodyksMarkramNeuron {
    inner: neurons::TsodyksMarkramNeuron,
}

#[pymethods]
impl PyTsodyksMarkramNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::TsodyksMarkramNeuron::new(),
        }
    }
    #[pyo3(signature = (current, presynaptic_spike=false))]
    fn step(&mut self, current: f64, presynaptic_spike: bool) -> i32 {
        self.inner.step(current, presynaptic_spike)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("x", self.inner.x)?;
        d.set_item("u", self.inner.u)?;
        Ok(d.into_any().unbind())
    }
}

py_neuron_default!("LiquidTimeConstantNeuron", PyLiquidTimeConstantNeuron, neurons::LiquidTimeConstantNeuron, state x);

// CompteWMNeuron: step(current, spike_in)
#[pyclass(
    name = "CompteWMNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyCompteWMNeuron {
    inner: neurons::CompteWMNeuron,
}

#[pymethods]
impl PyCompteWMNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::CompteWMNeuron::new(),
        }
    }
    #[pyo3(signature = (current, spike_in=false))]
    fn step(&mut self, current: f64, spike_in: bool) -> i32 {
        self.inner.step(current, spike_in)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("s_nmda", self.inner.s_nmda)?;
        Ok(d.into_any().unbind())
    }
}

// SiegertTransferFunction: step returns f64, no &mut self
#[pyclass(
    name = "SiegertTransferFunction",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySiegertTransferFunction {
    inner: neurons::SiegertTransferFunction,
}

#[pymethods]
impl PySiegertTransferFunction {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::SiegertTransferFunction::new(),
        }
    }
    fn step(&self, current: f64) -> f64 {
        self.inner.step(current)
    }
}

// ═══════════════════════════════════════════════════════════════════
// rate.rs models requiring non-default constructors or Vec state
// ═══════════════════════════════════════════════════════════════════

#[pyclass(
    name = "FractionalLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyFractionalLIFNeuron {
    inner: neurons::FractionalLIFNeuron,
}

#[pymethods]
impl PyFractionalLIFNeuron {
    #[new]
    #[pyo3(signature = (alpha=0.8, max_hist=50))]
    fn new(alpha: f64, max_hist: usize) -> Self {
        Self {
            inner: neurons::FractionalLIFNeuron::new(alpha, max_hist),
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
        d.set_item("v", self.inner.v)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "ParallelSpikingNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyParallelSpikingNeuron {
    inner: neurons::ParallelSpikingNeuron,
}

#[pymethods]
impl PyParallelSpikingNeuron {
    #[new]
    #[pyo3(signature = (kernel_size=8, v_threshold=1.0))]
    fn new(kernel_size: usize, v_threshold: f64) -> Self {
        Self {
            inner: neurons::ParallelSpikingNeuron::new(kernel_size, v_threshold),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pyclass(
    name = "AmariNeuralField",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAmariNeuralField {
    inner: neurons::AmariNeuralField,
}

#[pymethods]
impl PyAmariNeuralField {
    #[new]
    #[pyo3(signature = (n=64))]
    fn new(n: usize) -> Self {
        Self {
            inner: neurons::AmariNeuralField::new(n),
        }
    }
    fn step(&mut self, input: Vec<f64>) -> f64 {
        self.inner.step(&input)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.u.clone().into_pyarray(py)
    }
}

// ═══════════════════════════════════════════════════════════════════
// Registration function — call from lib.rs pymodule init
// ═══════════════════════════════════════════════════════════════════

pub fn register_neuron_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ai_optimized
    multi_timescale_neuron_binding::register(m)?;
    attention_gated_neuron_binding::register(m)?;
    predictive_coding_neuron_binding::register(m)?;
    self_referential_neuron_binding::register(m)?;
    compositional_binding_neuron_binding::register(m)?;
    differentiable_surrogate_neuron_binding::register(m)?;
    continuous_attractor_neuron_binding::register(m)?;
    meta_plastic_neuron_binding::register(m)?;
    // gap models (ai_optimized)
    adaptive_threshold_moe_neuron_binding::register(m)?;
    hybrid_linear_attention_neuron_binding::register(m)?;
    quantum_inspired_lif_neuron_binding::register(m)?;
    trivial_bindings::register(m)?;
    simple_spiking_bindings::register_primary(m)?;
    m.add_class::<PyBendaHerzNeuron>()?;
    m.add_class::<PyBrunelWangNeuron>()?;
    alpha_binding::register(m)?;
    simple_spiking_bindings::register_conductance_models(m)?;
    m.add_class::<PyEPropALIFNeuron>()?;
    m.add_class::<PySuperSpikeNeuron>()?;
    simple_spiking_bindings::register_tail(m)?;
    map_bindings::register(m)?;
    biophysical_bindings::register(m)?;
    multi_compartment_bindings::register(m)?;
    stochastic_bindings::register(m)?;
    m.add_class::<PyWilsonCowanUnit>()?;
    jansen_rit_binding::register(m)?;
    m.add_class::<PyWongWangUnit>()?;
    wong_wang_binding::register(m)?;
    ermentrout_kopell_pop_binding::register(m)?;
    m.add_class::<PyWendlingNeuron>()?;
    m.add_class::<PyLarterBreakspearNeuron>()?;
    hardware_bindings::register(m)?;
    // rate
    mcculloch_pitts_binding::register(m)?;
    sigmoid_rate_binding::register(m)?;
    threshold_linear_rate_binding::register(m)?;
    m.add_class::<PyAstrocyteModel>()?;
    m.add_class::<PyTsodyksMarkramNeuron>()?;
    m.add_class::<PyLiquidTimeConstantNeuron>()?;
    m.add_class::<PyCompteWMNeuron>()?;
    m.add_class::<PySiegertTransferFunction>()?;
    m.add_class::<PyFractionalLIFNeuron>()?;
    m.add_class::<PyParallelSpikingNeuron>()?;
    m.add_class::<PyAmariNeuralField>()?;
    m.add_class::<PyLeakyCompeteFireNeuron>()?;
    arcane_neuron_binding::register(m)?;
    // neuron.rs (legacy)
    adex_neuron_binding::register(m)?;
    crate::exp_if_binding::register_legacy_alias(m)?;
    lapicque_neuron_binding::register(m)?;
    interneuron_bindings::register(m)?;
    motor_bindings::register(m)?;
    sensory_bindings::register(m)?;
    cerebellar_bindings::register(m)?;
    channel_bindings::register(m)?;
    population_bindings::register(m)?;
    misc_bindings::register(m)?;
    synapse_bindings::register(m)?;
    Ok(())
}

// LeakyCompeteFireNeuron: step(Vec) -> Vec
#[pyclass(
    name = "LeakyCompeteFireNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLeakyCompeteFireNeuron {
    inner: neurons::LeakyCompeteFireNeuron,
}

#[pymethods]
impl PyLeakyCompeteFireNeuron {
    #[new]
    #[pyo3(signature = (n_units=4))]
    fn new(n_units: usize) -> Self {
        Self {
            inner: neurons::LeakyCompeteFireNeuron::new(n_units),
        }
    }
    fn step(&mut self, currents: Vec<f64>) -> Vec<i32> {
        self.inner.step(&currents)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v.clone())?;
        Ok(d.into_any().unbind())
    }
}
