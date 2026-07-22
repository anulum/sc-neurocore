// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 wrappers for all neuron models

//! PyO3 wrappers for all neuron models.
//!
//! Each wrapper follows the same pattern:
//!   #[pyclass(name = "Model")] struct Py<Model> { inner: neurons::<Model> }
//!   #[pymethods] impl Py<Model> { #[new] fn new(...) -> Self; fn step(...); fn reset(&mut self); fn get_state(...) }

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::neurons;

#[path = "bindings/adaptive_threshold_if.rs"]
mod adaptive_threshold_if_binding;
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
#[path = "bindings/resonate_and_fire.rs"]
mod resonate_and_fire_binding;
#[path = "bindings/sigmoid_rate.rs"]
mod sigmoid_rate_binding;
#[path = "bindings/threshold_linear_rate.rs"]
mod threshold_linear_rate_binding;
#[path = "bindings/wong_wang.rs"]
mod wong_wang_binding;

macro_rules! py_neuron_default {
    ($pylit:literal, $pyname:ident, $rust:ty $(, state $sname:ident)*) => {
        #[pyclass(name = $pylit, module = "sc_neurocore_engine.sc_neurocore_engine")]
        #[derive(Clone)]
        pub struct $pyname { inner: $rust }

        #[pymethods]
        impl $pyname {
            #[new]
            fn new() -> Self { Self { inner: <$rust>::default() } }

            fn step(&mut self, current: f64) -> i32 { self.inner.step(current) }

            fn reset(&mut self) { self.inner.reset(); }

            fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let d = PyDict::new(py);
                $(d.set_item(stringify!($sname), self.inner.$sname)?;)*
                Ok(d.into_any().unbind())
            }
        }
    };
}

#[path = "bindings/adaptive_threshold_moe_neuron.rs"]
mod adaptive_threshold_moe_neuron_binding;
#[path = "bindings/arcane_neuron.rs"]
mod arcane_neuron_binding;
#[path = "bindings/attention_gated_neuron.rs"]
mod attention_gated_neuron_binding;
#[path = "bindings/cerebellar/mod.rs"]
mod cerebellar_bindings;
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
#[path = "bindings/motor/mod.rs"]
mod motor_bindings;
#[path = "bindings/multi_timescale_neuron.rs"]
mod multi_timescale_neuron_binding;
#[path = "bindings/predictive_coding_neuron.rs"]
mod predictive_coding_neuron_binding;
#[path = "bindings/quantum_inspired_lif_neuron.rs"]
mod quantum_inspired_lif_neuron_binding;
#[path = "bindings/self_referential_neuron.rs"]
mod self_referential_neuron_binding;

// Gap models: CochlearHairCell (step(displacement) -> i32)
py_neuron_default!("RustCochlearHairCell", PyCochlearHairCell, neurons::CochlearHairCell, state v, state glutamate_release);

// Gap models: DendriticNMDANeuron (step(i_soma, glutamate))
#[pyclass(
    name = "RustDendriticNMDANeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyDendriticNMDANeuron {
    inner: neurons::DendriticNMDANeuron,
}

#[pymethods]
impl PyDendriticNMDANeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::DendriticNMDANeuron::new(),
        }
    }
    fn step(&mut self, i_soma: f64, glutamate: f64) -> i32 {
        self.inner.step(i_soma, glutamate)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v_soma", self.inner.v_soma)?;
        d.set_item("v_dend", self.inner.v_dend)?;
        Ok(d.into_any().unbind())
    }
}

// Gap models: MulticompartmentMCNNeuron (step_compartments(x_b, x_a, I))
#[pyclass(
    name = "RustMulticompartmentMCNNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyMulticompartmentMCNNeuron {
    inner: neurons::MulticompartmentMCNNeuron,
}

#[pymethods]
impl PyMulticompartmentMCNNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::MulticompartmentMCNNeuron::new(),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn step_compartments(&mut self, x_basal: f64, x_apical: f64, i_soma: f64) -> i32 {
        self.inner.step_compartments(x_basal, x_apical, i_soma)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("u", self.inner.u)?;
        d.set_item("v_basal", self.inner.v_basal)?;
        d.set_item("v_apical", self.inner.v_apical)?;
        Ok(d.into_any().unbind())
    }
}

// Gap models: AstrocyteLIFNeuron (step_with_pre(i_ext, pre_spike))
#[pyclass(
    name = "RustAstrocyteLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAstrocyteLIFNeuron {
    inner: neurons::AstrocyteLIFNeuron,
}

#[pymethods]
impl PyAstrocyteLIFNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::AstrocyteLIFNeuron::new(),
        }
    }
    fn step(&mut self, current: f64) -> i32 {
        self.inner.step(current)
    }
    fn step_with_pre(&mut self, i_ext: f64, pre_spike: bool) -> i32 {
        self.inner.step_with_pre(i_ext, pre_spike)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("ca", self.inner.ca)?;
        Ok(d.into_any().unbind())
    }
}

// Gap models: DirectionSelectiveRGC (step_rf(intensity, surround))
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

// Gap synapse models: TripletStdpSynapse
#[pyclass(
    name = "RustTripletStdpSynapse",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTripletStdpSynapse {
    inner: crate::synapses::TripletStdpSynapse,
}

#[pymethods]
impl PyTripletStdpSynapse {
    #[new]
    #[pyo3(signature = (weight=0.5, w_min=0.0, w_max=1.0))]
    fn new(weight: f64, w_min: f64, w_max: f64) -> Self {
        Self {
            inner: crate::synapses::TripletStdpSynapse::new(weight, w_min, w_max),
        }
    }
    fn step(&mut self, pre_spike: bool, post_spike: bool) {
        self.inner.step(pre_spike, post_spike);
    }
    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("weight", self.inner.weight)?;
        d.set_item("r1", self.inner.r1)?;
        d.set_item("o1", self.inner.o1)?;
        d.set_item("r2", self.inner.r2)?;
        d.set_item("o2", self.inner.o2)?;
        Ok(d.into_any().unbind())
    }
}

// Gap synapse models: ShortTermPlasticitySynapse
#[pyclass(
    name = "RustShortTermPlasticitySynapse",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyShortTermPlasticitySynapse {
    inner: crate::synapses::ShortTermPlasticitySynapse,
}

#[pymethods]
impl PyShortTermPlasticitySynapse {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::synapses::ShortTermPlasticitySynapse::new_depressing(),
        }
    }
    #[staticmethod]
    fn depressing() -> Self {
        Self {
            inner: crate::synapses::ShortTermPlasticitySynapse::new_depressing(),
        }
    }
    #[staticmethod]
    fn facilitating() -> Self {
        Self {
            inner: crate::synapses::ShortTermPlasticitySynapse::new_facilitating(),
        }
    }
    fn step(&mut self, pre_spike: bool) -> f64 {
        self.inner.step(pre_spike)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("x", self.inner.x)?;
        d.set_item("u", self.inner.u)?;
        Ok(d.into_any().unbind())
    }
}

// Gap synapse models: DopamineStdpSynapse
#[pyclass(
    name = "RustDopamineStdpSynapse",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyDopamineStdpSynapse {
    inner: crate::synapses::DopamineStdpSynapse,
}

#[pymethods]
impl PyDopamineStdpSynapse {
    #[new]
    #[pyo3(signature = (weight=0.5, w_min=0.0, w_max=1.0))]
    fn new(weight: f64, w_min: f64, w_max: f64) -> Self {
        Self {
            inner: crate::synapses::DopamineStdpSynapse::new(weight, w_min, w_max),
        }
    }
    fn step(&mut self, pre_spike: bool, post_spike: bool, reward: f64) {
        self.inner.step(pre_spike, post_spike, reward);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }
    #[getter]
    fn dopamine(&self) -> f64 {
        self.inner.dopamine
    }
    #[getter]
    fn eligibility(&self) -> f64 {
        self.inner.eligibility
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("weight", self.inner.weight)?;
        d.set_item("eligibility", self.inner.eligibility)?;
        d.set_item("dopamine", self.inner.dopamine)?;
        d.set_item("trace_pre", self.inner.trace_pre)?;
        d.set_item("trace_post", self.inner.trace_post)?;
        Ok(d.into_any().unbind())
    }
}

// ═══════════════════════════════════════════════════════════════════
// trivial.rs models
// ═══════════════════════════════════════════════════════════════════

py_neuron_default!("QuadraticIFNeuron", PyQuadraticIFNeuron, neurons::QuadraticIFNeuron, state v);
py_neuron_default!("ThetaNeuron", PyThetaNeuron, neurons::ThetaNeuron, state theta);
py_neuron_default!("PerfectIntegratorNeuron", PyPerfectIntegratorNeuron, neurons::PerfectIntegratorNeuron, state v);
py_neuron_default!("GatedLIFNeuron", PyGatedLIFNeuron, neurons::GatedLIFNeuron, state v);
py_neuron_default!("NonlinearLIFNeuron", PyNonlinearLIFNeuron, neurons::NonlinearLIFNeuron, state v, state w);
py_neuron_default!("SFANeuron", PySFANeuron, neurons::SFANeuron, state v, state g_sfa);
py_neuron_default!("MATNeuron", PyMATNeuron, neurons::MATNeuron, state v, state theta1, state theta2);
py_neuron_default!("KLIFNeuron", PyKLIFNeuron, neurons::KLIFNeuron, state v);
py_neuron_default!("InhibitoryLIFNeuron", PyInhibitoryLIFNeuron, neurons::InhibitoryLIFNeuron, state v, state inh_trace);
py_neuron_default!("ComplementaryLIFNeuron", PyComplementaryLIFNeuron, neurons::ComplementaryLIFNeuron, state v_pos, state v_neg);
py_neuron_default!("ParametricLIFNeuron", PyParametricLIFNeuron, neurons::ParametricLIFNeuron, state v);
py_neuron_default!("NonResettingLIFNeuron", PyNonResettingLIFNeuron, neurons::NonResettingLIFNeuron, state v, state theta);
py_neuron_default!("AdaptiveThresholdIFNeuron", PyAdaptiveThresholdIFNeuron, neurons::AdaptiveThresholdIFNeuron, state v, state theta);
py_neuron_default!("SigmaDeltaNeuron", PySigmaDeltaNeuron, neurons::SigmaDeltaNeuron, state sigma);
py_neuron_default!("EnergyLIFNeuron", PyEnergyLIFNeuron, neurons::EnergyLIFNeuron, state v, state epsilon);
py_neuron_default!("ClosedFormContinuousNeuron", PyClosedFormContinuousNeuron, neurons::ClosedFormContinuousNeuron, state x);

// ═══════════════════════════════════════════════════════════════════
// simple_spiking.rs models
// ═══════════════════════════════════════════════════════════════════

py_neuron_default!("FitzHughNagumoNeuron", PyFitzHughNagumoNeuron, neurons::FitzHughNagumoNeuron, state v, state w);
py_neuron_default!("MorrisLecarNeuron", PyMorrisLecarNeuron, neurons::MorrisLecarNeuron, state v, state w);
py_neuron_default!("HindmarshRoseNeuron", PyHindmarshRoseNeuron, neurons::HindmarshRoseNeuron, state x, state y, state z);
py_neuron_default!("ResonateAndFireNeuron", PyResonateAndFireNeuron, neurons::ResonateAndFireNeuron, state x, state y);
py_neuron_default!("BalancedResonateAndFireNeuron", PyBalancedResonateAndFireNeuron, neurons::BalancedResonateAndFireNeuron, state x, state y, state q);
py_neuron_default!("FitzHughRinzelNeuron", PyFitzHughRinzelNeuron, neurons::FitzHughRinzelNeuron, state v, state w, state y);
py_neuron_default!("McKeanNeuron", PyMcKeanNeuron, neurons::McKeanNeuron, state v, state w);
py_neuron_default!("TermanWangOscillator", PyTermanWangOscillator, neurons::TermanWangOscillator, state v, state w);
py_neuron_default!("GutkinErmentroutNeuron", PyGutkinErmentroutNeuron, neurons::GutkinErmentroutNeuron, state v, state n);
py_neuron_default!("WilsonHRNeuron", PyWilsonHRNeuron, neurons::WilsonHRNeuron, state v, state r);
py_neuron_default!("ChayNeuron", PyChayNeuron, neurons::ChayNeuron, state v, state n, state ca);
py_neuron_default!("ChayKeizerNeuron", PyChayKeizerNeuron, neurons::ChayKeizerNeuron, state v, state n, state ca);
py_neuron_default!("ShermanRinzelKeizerNeuron", PyShermanRinzelKeizerNeuron, neurons::ShermanRinzelKeizerNeuron, state v, state n, state s);
py_neuron_default!("ButeraRespiratoryNeuron", PyButeraRespiratoryNeuron, neurons::ButeraRespiratoryNeuron, state v, state n, state h_nap);
py_neuron_default!("LearnableNeuronModel", PyLearnableNeuronModel, neurons::LearnableNeuronModel, state v);
py_neuron_default!("PernarowskiNeuron", PyPernarowskiNeuron, neurons::PernarowskiNeuron, state v, state w, state z);

// EPropALIFNeuron: needs tau params
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

// SuperSpikeNeuron: needs tau params
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

// BendaHerzNeuron: needs seed
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

// ═══════════════════════════════════════════════════════════════════
// maps.rs models
// ═══════════════════════════════════════════════════════════════════

py_neuron_default!("ChialvoMapNeuron", PyChialvoMapNeuron, neurons::ChialvoMapNeuron, state x, state y);
py_neuron_default!("RulkovMapNeuron", PyRulkovMapNeuron, neurons::RulkovMapNeuron, state x, state y);
py_neuron_default!("IbarzTanakaMapNeuron", PyIbarzTanakaMapNeuron, neurons::IbarzTanakaMapNeuron, state v, state u);
py_neuron_default!("MedvedevMapNeuron", PyMedvedevMapNeuron, neurons::MedvedevMapNeuron, state u);
py_neuron_default!("CazellesMapNeuron", PyCazellesMapNeuron, neurons::CazellesMapNeuron, state x, state y);
py_neuron_default!("CourageNekorkinMapNeuron", PyCourageNekorkinMapNeuron, neurons::CourageNekorkinMapNeuron, state x, state y);
py_neuron_default!("AiharaMapNeuron", PyAiharaMapNeuron, neurons::AiharaMapNeuron, state x, state y);
py_neuron_default!("KilincBhattMapNeuron", PyKilincBhattMapNeuron, neurons::KilincBhattMapNeuron, state x, state theta);
py_neuron_default!("ErmentroutKopellMapNeuron", PyErmentroutKopellMapNeuron, neurons::ErmentroutKopellMapNeuron, state theta);

// ═══════════════════════════════════════════════════════════════════
// biophysical/ model modules
// ═══════════════════════════════════════════════════════════════════

py_neuron_default!("HodgkinHuxleyNeuron", PyHodgkinHuxleyNeuron, neurons::HodgkinHuxleyNeuron, state v, state m, state h, state n);
py_neuron_default!("TraubMilesNeuron", PyTraubMilesNeuron, neurons::TraubMilesNeuron, state v, state m, state h, state n);
py_neuron_default!("WangBuzsakiNeuron", PyWangBuzsakiNeuron, neurons::WangBuzsakiNeuron, state v, state h, state n);
py_neuron_default!("ConnorStevensNeuron", PyConnorStevensNeuron, neurons::ConnorStevensNeuron, state v, state m, state h, state n, state a, state b);
py_neuron_default!("DestexheThalamicNeuron", PyDestexheThalamicNeuron, neurons::DestexheThalamicNeuron, state v, state h_na, state n_k, state m_t, state h_t);
py_neuron_default!("HuberBraunNeuron", PyHuberBraunNeuron, neurons::HuberBraunNeuron, state v, state a_sd, state a_sr);
py_neuron_default!("GolombFSNeuron", PyGolombFSNeuron, neurons::GolombFSNeuron, state v, state h, state n, state p);
py_neuron_default!("PospischilNeuron", PyPospischilNeuron, neurons::PospischilNeuron, state v, state m, state h, state n, state p);
py_neuron_default!("MainenSejnowskiNeuron", PyMainenSejnowskiNeuron, neurons::MainenSejnowskiNeuron, state vs, state va, state m, state h, state n);
py_neuron_default!("DeSchutterPurkinjeNeuron", PyDeSchutterPurkinjeNeuron, neurons::DeSchutterPurkinjeNeuron, state v, state h_na, state n_k, state m_cap, state h_cap, state q_kca, state ca);
py_neuron_default!("PlantR15Neuron", PyPlantR15Neuron, neurons::PlantR15Neuron, state v, state m, state h, state n, state ca);
py_neuron_default!("PrescottNeuron", PyPrescottNeuron, neurons::PrescottNeuron, state v, state w);
py_neuron_default!("MihalasNieburNeuron", PyMihalasNieburNeuron, neurons::MihalasNieburNeuron, state v, state theta, state i1, state i2);
py_neuron_default!("GLIFNeuron", PyGLIFNeuron, neurons::GLIFNeuron, state v, state theta, state i_asc1, state i_asc2);
py_neuron_default!("AvRonCardiacNeuron", PyAvRonCardiacNeuron, neurons::AvRonCardiacNeuron, state v, state h, state n, state s);
py_neuron_default!("DurstewitzDopamineNeuron", PyDurstewitzDopamineNeuron, neurons::DurstewitzDopamineNeuron, state v, state h_na, state n_k);
py_neuron_default!("HillTononiNeuron", PyHillTononiNeuron, neurons::HillTononiNeuron, state v, state h_na, state n_k, state m_h, state h_t, state na_i);
py_neuron_default!("BertramPhantomBurster", PyBertramPhantomBurster, neurons::BertramPhantomBurster, state v, state s1, state s2);
py_neuron_default!("YamadaNeuron", PyYamadaNeuron, neurons::YamadaNeuron, state v, state n, state q);

// GIFPopulationNeuron: needs seed
#[pyclass(
    name = "GIFPopulationNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyGIFPopulationNeuron {
    inner: neurons::GIFPopulationNeuron,
}

#[pymethods]
impl PyGIFPopulationNeuron {
    #[new]
    #[pyo3(signature = (seed=42))]
    fn new(seed: u64) -> Self {
        Self {
            inner: neurons::GIFPopulationNeuron::new(seed),
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
        d.set_item("theta", self.inner.theta)?;
        d.set_item("eta", self.inner.eta)?;
        Ok(d.into_any().unbind())
    }
}

// ═══════════════════════════════════════════════════════════════════
// multi_compartment.rs models
// ═══════════════════════════════════════════════════════════════════

// PinskyRinzelNeuron: step(current_soma, current_dend)
#[pyclass(
    name = "PinskyRinzelNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyPinskyRinzelNeuron {
    inner: neurons::PinskyRinzelNeuron,
}

#[pymethods]
impl PyPinskyRinzelNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::PinskyRinzelNeuron::new(),
        }
    }
    #[pyo3(signature = (current_soma, current_dend=0.0))]
    fn step(&mut self, current_soma: f64, current_dend: f64) -> i32 {
        self.inner.step(current_soma, current_dend)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v_s", self.inner.v_s)?;
        d.set_item("v_d", self.inner.v_d)?;
        Ok(d.into_any().unbind())
    }
}

// HayL5PyramidalNeuron: step(current_soma, current_tuft)
#[pyclass(
    name = "HayL5PyramidalNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyHayL5PyramidalNeuron {
    inner: neurons::HayL5PyramidalNeuron,
}

#[pymethods]
impl PyHayL5PyramidalNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::HayL5PyramidalNeuron::new(),
        }
    }
    #[pyo3(signature = (current_soma, current_tuft=0.0))]
    fn step(&mut self, current_soma: f64, current_tuft: f64) -> i32 {
        self.inner.step(current_soma, current_tuft)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v_s", self.inner.v_s)?;
        d.set_item("v_t", self.inner.v_t)?;
        d.set_item("v_a", self.inner.v_a)?;
        Ok(d.into_any().unbind())
    }
}

py_neuron_default!("MarderSTGNeuron", PyMarderSTGNeuron, neurons::MarderSTGNeuron, state v, state ca);
py_neuron_default!("BoothRinzelNeuron", PyBoothRinzelNeuron, neurons::BoothRinzelNeuron, state vs, state vd, state ca);
py_neuron_default!("DendrifyNeuron", PyDendrifyNeuron, neurons::DendrifyNeuron, state v_s, state v_d);

// RallCableNeuron: variable compartments
#[pyclass(
    name = "RallCableNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyRallCableNeuron {
    inner: neurons::RallCableNeuron,
}

#[pymethods]
impl PyRallCableNeuron {
    #[new]
    #[pyo3(signature = (n_comp=5))]
    fn new(n_comp: usize) -> Self {
        Self {
            inner: neurons::RallCableNeuron::new(n_comp),
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
        d.set_item("v", self.inner.v.clone())?;
        Ok(d.into_any().unbind())
    }
}

// TwoCompartmentLIFNeuron: step(i_soma, i_dend)
#[pyclass(
    name = "TwoCompartmentLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTwoCompartmentLIFNeuron {
    inner: neurons::TwoCompartmentLIFNeuron,
}

#[pymethods]
impl PyTwoCompartmentLIFNeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::TwoCompartmentLIFNeuron::new(),
        }
    }
    #[pyo3(signature = (i_soma, i_dend=0.0))]
    fn step(&mut self, i_soma: f64, i_dend: f64) -> i32 {
        self.inner.step(i_soma, i_dend)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v_s", self.inner.v_s)?;
        d.set_item("v_d", self.inner.v_d)?;
        Ok(d.into_any().unbind())
    }
}

// ═══════════════════════════════════════════════════════════════════
// special.rs models (stochastic / population / neural mass)
// ═══════════════════════════════════════════════════════════════════

#[pyclass(
    name = "InhomogeneousPoissonNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyInhomogeneousPoissonNeuron {
    inner: neurons::InhomogeneousPoissonNeuron,
}

#[pymethods]
impl PyInhomogeneousPoissonNeuron {
    #[new]
    #[pyo3(signature = (dt_ms=1.0, seed=42))]
    fn new(dt_ms: f64, seed: u64) -> Self {
        Self {
            inner: neurons::InhomogeneousPoissonNeuron::new(dt_ms, seed),
        }
    }
    fn step(&mut self, rate_hz: f64) -> i32 {
        self.inner.step(rate_hz)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pyclass(
    name = "GammaRenewalNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyGammaRenewalNeuron {
    inner: neurons::GammaRenewalNeuron,
}

#[pymethods]
impl PyGammaRenewalNeuron {
    #[new]
    #[pyo3(signature = (rate_hz=50.0, shape_k=3, seed=42))]
    fn new(rate_hz: f64, shape_k: u32, seed: u64) -> Self {
        Self {
            inner: neurons::GammaRenewalNeuron::new(rate_hz, shape_k, seed),
        }
    }
    #[pyo3(signature = (rate_override=-1.0))]
    fn step(&mut self, rate_override: f64) -> i32 {
        self.inner.step(rate_override)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pyclass(
    name = "StochasticIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyStochasticIFNeuron {
    inner: neurons::StochasticIFNeuron,
}

#[pymethods]
impl PyStochasticIFNeuron {
    #[new]
    #[pyo3(signature = (seed=42))]
    fn new(seed: u64) -> Self {
        Self {
            inner: neurons::StochasticIFNeuron::new(seed),
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
    name = "StochasticLIFNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyStochasticLIFNeuron {
    inner: neurons::StochasticLIFNeuron,
}

#[pymethods]
impl PyStochasticLIFNeuron {
    #[new]
    #[pyo3(signature = (seed=42))]
    fn new(seed: u64) -> Self {
        Self {
            inner: neurons::StochasticLIFNeuron::new(seed),
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
    name = "GalvesLocherbachNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyGalvesLocherbachNeuron {
    inner: neurons::GalvesLocherbachNeuron,
}

#[pymethods]
impl PyGalvesLocherbachNeuron {
    #[new]
    #[pyo3(signature = (seed=42))]
    fn new(seed: u64) -> Self {
        Self {
            inner: neurons::GalvesLocherbachNeuron::new(seed),
        }
    }
    fn step(&mut self, weighted_input: f64) -> i32 {
        self.inner.step(weighted_input)
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

py_neuron_default!("SpikeResponseNeuron", PySpikeResponseNeuron, neurons::SpikeResponseNeuron, state v, state time_since_spike);

// GLMNeuron: needs n_k, n_h, seed
#[pyclass(name = "GLMNeuron", module = "sc_neurocore_engine.sc_neurocore_engine")]
#[derive(Clone)]
pub struct PyGLMNeuron {
    inner: neurons::GLMNeuron,
}

#[pymethods]
impl PyGLMNeuron {
    #[new]
    #[pyo3(signature = (n_k=10, n_h=20, seed=42))]
    fn new(n_k: usize, n_h: usize, seed: u64) -> Self {
        Self {
            inner: neurons::GLMNeuron::new(n_k, n_h, seed),
        }
    }
    fn step(&mut self, stimulus: f64) -> i32 {
        self.inner.step(stimulus)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

// WilsonCowanUnit: step returns f64
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

// WongWangUnit: step(stim1, stim2) -> (f64, f64)
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

// WendlingNeuron: step returns f64
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

// LarterBreakspearNeuron: step returns f64
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
// hardware.rs models
// ═══════════════════════════════════════════════════════════════════

#[pyclass(
    name = "LoihiCUBANeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLoihiCUBANeuron {
    inner: neurons::LoihiCUBANeuron,
}

#[pymethods]
impl PyLoihiCUBANeuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::LoihiCUBANeuron::new(),
        }
    }
    fn step(&mut self, weighted_input: i32) -> i32 {
        self.inner.step(weighted_input)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("u", self.inner.u)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "Loihi2Neuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyLoihi2Neuron {
    inner: neurons::Loihi2Neuron,
}

#[pymethods]
impl PyLoihi2Neuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::Loihi2Neuron::new(),
        }
    }
    fn step(&mut self, weighted_input: i32) -> i32 {
        self.inner.step(weighted_input)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("s1", self.inner.s1)?;
        d.set_item("s2", self.inner.s2)?;
        d.set_item("s3", self.inner.s3)?;
        Ok(d.into_any().unbind())
    }
}

#[pyclass(
    name = "TrueNorthNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyTrueNorthNeuron {
    inner: neurons::TrueNorthNeuron,
}

#[pymethods]
impl PyTrueNorthNeuron {
    #[new]
    #[pyo3(signature = (threshold=100))]
    fn new(threshold: i32) -> Self {
        Self {
            inner: neurons::TrueNorthNeuron::new(threshold),
        }
    }
    fn step(&mut self, weighted_input: i32) -> i32 {
        self.inner.step(weighted_input)
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

py_neuron_default!("BrainScaleSAdExNeuron", PyBrainScaleSAdExNeuron, neurons::BrainScaleSAdExNeuron, state v, state w);
py_neuron_default!("SpiNNakerLIFNeuron", PySpiNNakerLIFNeuron, neurons::SpiNNakerLIFNeuron, state v, state refrac_count);

#[pyclass(
    name = "SpiNNaker2Neuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PySpiNNaker2Neuron {
    inner: neurons::SpiNNaker2Neuron,
}

#[pymethods]
impl PySpiNNaker2Neuron {
    #[new]
    fn new() -> Self {
        Self {
            inner: neurons::SpiNNaker2Neuron::new(),
        }
    }
    fn step(&mut self, current: i32) -> i32 {
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

py_neuron_default!(
    "DPINeuron",
    PyDPINeuron,
    neurons::DPINeuron,
    state i_mem,
    state i_ahp,
    state refractory_time
);

#[pyclass(
    name = "AkidaNeuron",
    module = "sc_neurocore_engine.sc_neurocore_engine"
)]
#[derive(Clone)]
pub struct PyAkidaNeuron {
    inner: neurons::AkidaNeuron,
}

#[pymethods]
impl PyAkidaNeuron {
    #[new]
    #[pyo3(signature = (threshold=100))]
    fn new(threshold: i32) -> Self {
        Self {
            inner: neurons::AkidaNeuron::new(threshold),
        }
    }
    fn step(&mut self, weight: i32) -> i32 {
        self.inner.step(weight as f64)
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let d = PyDict::new(py);
        d.set_item("v", self.inner.v)?;
        d.set_item("rank", self.inner.rank)?;
        Ok(d.into_any().unbind())
    }
}

py_neuron_default!("NeuroGridNeuron", PyNeuroGridNeuron, neurons::NeuroGridNeuron, state v_s, state v_d);

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

// channels.rs models
py_neuron_default!("PersistentNaNeuron", PyPersistentNaNeuron, neurons::PersistentNaNeuron, state v, state h, state n, state p);
py_neuron_default!("IhNeuron", PyIhNeuron, neurons::IhNeuron, state v, state h, state n, state r);
py_neuron_default!("TTypeCaNeuron", PyTTypeCaNeuron, neurons::TTypeCaNeuron, state v, state h, state n, state s);
py_neuron_default!("ATypeKNeuron", PyATypeKNeuron, neurons::ATypeKNeuron, state v, state h, state n, state a, state b);
py_neuron_default!("BKNeuron", PyBKNeuron, neurons::BKNeuron, state v, state h, state n, state ca);
py_neuron_default!("SKNeuron", PySKNeuron, neurons::SKNeuron, state v, state h, state n, state ca);
py_neuron_default!("NMDANeuron", PyNMDANeuron, neurons::NMDANeuron, state v, state h, state n, state s_nmda);

// population.rs models
py_neuron_default!("MontbrioMeanField", PyMontbrioMeanField, neurons::MontbrioMeanField, state r, state v);
py_neuron_default!("BrunelNetwork", PyBrunelNetwork, neurons::BrunelNetwork, state r_e, state r_i);
py_neuron_default!("TUMNetwork", PyTUMNetwork, neurons::TUMNetwork, state r, state x, state u);
py_neuron_default!("ElBoustaniNetwork", PyElBoustaniNetwork, neurons::ElBoustaniNetwork, state r_e, state r_i, state s);

// misc.rs models
py_neuron_default!("GradedSynapseNeuron", PyGradedSynapseNeuron, neurons::GradedSynapseNeuron, state v);
py_neuron_default!("GapJunctionNeuron", PyGapJunctionNeuron, neurons::GapJunctionNeuron, state v);
py_neuron_default!("FrankenhaeUserHuxleyAxon", PyFHAxon, neurons::FrankenhaeUserHuxleyAxon, state v, state m, state h, state n, state p);
py_neuron_default!("NodeOfRanvier", PyNodeOfRanvier, neurons::NodeOfRanvier, state v, state m, state h, state p, state s);
py_neuron_default!("MyelinatedAxon", PyMyelinatedAxon, neurons::MyelinatedAxon, state v_inter);
py_neuron_default!("CardiacPurkinjeFibre", PyCardiacPurkinjeFibre, neurons::CardiacPurkinjeFibre, state v, state d, state f, state y);
py_neuron_default!("SmoothMuscleCell", PySmoothMuscleCell, neurons::SmoothMuscleCell, state v, state ca, state ca_store);
py_neuron_default!("EndocrineBetaCell", PyEndocrineBetaCell, neurons::EndocrineBetaCell, state v, state n, state ca);

// ═══════════════════════════════════════════════════════════════════
// sensory.rs models (10 sensory neuron types)
// ═══════════════════════════════════════════════════════════════════

// Graded sensory neurons (step returns f64)
macro_rules! py_sensory_graded {
    ($pylit:literal, $pyname:ident, $rust:ty $(, state $sname:ident)*) => {
        #[pyclass(name = $pylit, module = "sc_neurocore_engine.sc_neurocore_engine")]
        #[derive(Clone)]
        pub struct $pyname { inner: $rust }

        #[pymethods]
        impl $pyname {
            #[new]
            fn new() -> Self { Self { inner: <$rust>::default() } }

            fn step(&mut self, input: f64) -> f64 { self.inner.step(input) }

            fn reset(&mut self) { self.inner.reset(); }

            fn get_state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let d = PyDict::new(py);
                $(d.set_item(stringify!($sname), self.inner.$sname)?;)*
                Ok(d.into_any().unbind())
            }
        }
    };
}

py_sensory_graded!("InnerHairCell", PyInnerHairCell, neurons::InnerHairCell, state v, state ca, state q, state c, state w);
py_sensory_graded!("OuterHairCell", PyOuterHairCell, neurons::OuterHairCell, state v, state motility);
py_sensory_graded!("RodPhotoreceptor", PyRodPhotoreceptor, neurons::RodPhotoreceptor, state v, state cgmp, state ca);
py_sensory_graded!("ConePhotoreceptor", PyConePhotoreceptor, neurons::ConePhotoreceptor, state v, state cgmp);
py_sensory_graded!("TasteReceptorCell", PyTasteReceptorCell, neurons::TasteReceptorCell, state v, state ca, state ip3, state atp_release);

// Spiking sensory neurons (step returns i32)
py_neuron_default!("MerkelCell", PyMerkelCell, neurons::MerkelCell, state v, state adapt);
py_neuron_default!("Nociceptor", PyNociceptor, neurons::Nociceptor, state v, state sensitisation);
py_neuron_default!("OlfactoryReceptorNeuron", PyOlfactoryReceptorNeuron, neurons::OlfactoryReceptorNeuron, state v, state camp, state adapt, state pde4);

// RetinalGanglionCell and PacinianCorpuscle: spiking but non-default constructors
py_neuron_default!("RetinalGanglionCell", PyRetinalGanglionCell, neurons::RetinalGanglionCell, state baseline, state on_centre);
py_neuron_default!("PacinianCorpuscle", PyPacinianCorpuscle, neurons::PacinianCorpuscle, state v, state prev_pressure, state adapt);

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
    // trivial
    m.add_class::<PyQuadraticIFNeuron>()?;
    m.add_class::<PyThetaNeuron>()?;
    m.add_class::<PyPerfectIntegratorNeuron>()?;
    m.add_class::<PyGatedLIFNeuron>()?;
    m.add_class::<PyNonlinearLIFNeuron>()?;
    m.add_class::<PySFANeuron>()?;
    m.add_class::<PyMATNeuron>()?;
    m.add_class::<PyKLIFNeuron>()?;
    m.add_class::<PyInhibitoryLIFNeuron>()?;
    m.add_class::<PyComplementaryLIFNeuron>()?;
    m.add_class::<PyParametricLIFNeuron>()?;
    m.add_class::<PyNonResettingLIFNeuron>()?;
    m.add_class::<PyAdaptiveThresholdIFNeuron>()?;
    adaptive_threshold_if_binding::register(m)?;
    m.add_class::<PySigmaDeltaNeuron>()?;
    m.add_class::<PyEnergyLIFNeuron>()?;
    m.add_class::<PyClosedFormContinuousNeuron>()?;
    // simple_spiking
    m.add_class::<PyFitzHughNagumoNeuron>()?;
    m.add_class::<PyMorrisLecarNeuron>()?;
    m.add_class::<PyHindmarshRoseNeuron>()?;
    m.add_class::<PyResonateAndFireNeuron>()?;
    resonate_and_fire_binding::register(m)?;
    m.add_class::<PyBalancedResonateAndFireNeuron>()?;
    m.add_class::<PyFitzHughRinzelNeuron>()?;
    m.add_class::<PyMcKeanNeuron>()?;
    m.add_class::<PyTermanWangOscillator>()?;
    m.add_class::<PyBendaHerzNeuron>()?;
    m.add_class::<PyBrunelWangNeuron>()?;
    alpha_binding::register(m)?;
    m.add_class::<PyGutkinErmentroutNeuron>()?;
    m.add_class::<PyWilsonHRNeuron>()?;
    m.add_class::<PyChayNeuron>()?;
    m.add_class::<PyChayKeizerNeuron>()?;
    m.add_class::<PyShermanRinzelKeizerNeuron>()?;
    m.add_class::<PyButeraRespiratoryNeuron>()?;
    m.add_class::<PyEPropALIFNeuron>()?;
    m.add_class::<PySuperSpikeNeuron>()?;
    m.add_class::<PyLearnableNeuronModel>()?;
    m.add_class::<PyPernarowskiNeuron>()?;
    // maps
    m.add_class::<PyChialvoMapNeuron>()?;
    m.add_class::<PyRulkovMapNeuron>()?;
    m.add_class::<PyIbarzTanakaMapNeuron>()?;
    m.add_class::<PyMedvedevMapNeuron>()?;
    m.add_class::<PyCazellesMapNeuron>()?;
    m.add_class::<PyCourageNekorkinMapNeuron>()?;
    m.add_class::<PyAiharaMapNeuron>()?;
    m.add_class::<PyKilincBhattMapNeuron>()?;
    m.add_class::<PyErmentroutKopellMapNeuron>()?;
    // biophysical
    m.add_class::<PyHodgkinHuxleyNeuron>()?;
    m.add_class::<PyTraubMilesNeuron>()?;
    m.add_class::<PyWangBuzsakiNeuron>()?;
    m.add_class::<PyConnorStevensNeuron>()?;
    m.add_class::<PyDestexheThalamicNeuron>()?;
    m.add_class::<PyHuberBraunNeuron>()?;
    m.add_class::<PyGolombFSNeuron>()?;
    m.add_class::<PyPospischilNeuron>()?;
    m.add_class::<PyMainenSejnowskiNeuron>()?;
    m.add_class::<PyDeSchutterPurkinjeNeuron>()?;
    m.add_class::<PyPlantR15Neuron>()?;
    m.add_class::<PyPrescottNeuron>()?;
    m.add_class::<PyMihalasNieburNeuron>()?;
    m.add_class::<PyGLIFNeuron>()?;
    m.add_class::<PyGIFPopulationNeuron>()?;
    m.add_class::<PyAvRonCardiacNeuron>()?;
    m.add_class::<PyDurstewitzDopamineNeuron>()?;
    m.add_class::<PyHillTononiNeuron>()?;
    m.add_class::<PyBertramPhantomBurster>()?;
    m.add_class::<PyYamadaNeuron>()?;
    // multi_compartment
    m.add_class::<PyPinskyRinzelNeuron>()?;
    m.add_class::<PyHayL5PyramidalNeuron>()?;
    m.add_class::<PyMarderSTGNeuron>()?;
    m.add_class::<PyRallCableNeuron>()?;
    m.add_class::<PyBoothRinzelNeuron>()?;
    m.add_class::<PyDendrifyNeuron>()?;
    m.add_class::<PyTwoCompartmentLIFNeuron>()?;
    // gap models (multi_compartment)
    m.add_class::<PyDendriticNMDANeuron>()?;
    m.add_class::<PyMulticompartmentMCNNeuron>()?;
    m.add_class::<PyAstrocyteLIFNeuron>()?;
    // special
    m.add_class::<PyInhomogeneousPoissonNeuron>()?;
    m.add_class::<PyGammaRenewalNeuron>()?;
    m.add_class::<PyStochasticIFNeuron>()?;
    m.add_class::<PyStochasticLIFNeuron>()?;
    m.add_class::<PyGalvesLocherbachNeuron>()?;
    m.add_class::<PySpikeResponseNeuron>()?;
    m.add_class::<PyGLMNeuron>()?;
    m.add_class::<PyWilsonCowanUnit>()?;
    jansen_rit_binding::register(m)?;
    m.add_class::<PyWongWangUnit>()?;
    wong_wang_binding::register(m)?;
    ermentrout_kopell_pop_binding::register(m)?;
    m.add_class::<PyWendlingNeuron>()?;
    m.add_class::<PyLarterBreakspearNeuron>()?;
    // hardware
    m.add_class::<PyLoihiCUBANeuron>()?;
    m.add_class::<PyLoihi2Neuron>()?;
    m.add_class::<PyTrueNorthNeuron>()?;
    m.add_class::<PyBrainScaleSAdExNeuron>()?;
    m.add_class::<PySpiNNakerLIFNeuron>()?;
    m.add_class::<PySpiNNaker2Neuron>()?;
    m.add_class::<PyDPINeuron>()?;
    m.add_class::<PyAkidaNeuron>()?;
    m.add_class::<PyNeuroGridNeuron>()?;
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
    // interneurons
    interneuron_bindings::register(m)?;
    // motor
    motor_bindings::register(m)?;
    // sensory
    m.add_class::<PyInnerHairCell>()?;
    m.add_class::<PyOuterHairCell>()?;
    m.add_class::<PyRodPhotoreceptor>()?;
    m.add_class::<PyConePhotoreceptor>()?;
    m.add_class::<PyRetinalGanglionCell>()?;
    m.add_class::<PyMerkelCell>()?;
    m.add_class::<PyPacinianCorpuscle>()?;
    m.add_class::<PyNociceptor>()?;
    m.add_class::<PyOlfactoryReceptorNeuron>()?;
    m.add_class::<PyTasteReceptorCell>()?;
    // gap models (sensory)
    m.add_class::<PyDirectionSelectiveRGC>()?;
    m.add_class::<PyCochlearHairCell>()?;
    // cerebellar
    cerebellar_bindings::register(m)?;
    // channels
    m.add_class::<PyPersistentNaNeuron>()?;
    m.add_class::<PyIhNeuron>()?;
    m.add_class::<PyTTypeCaNeuron>()?;
    m.add_class::<PyATypeKNeuron>()?;
    m.add_class::<PyBKNeuron>()?;
    m.add_class::<PySKNeuron>()?;
    m.add_class::<PyNMDANeuron>()?;
    // population
    m.add_class::<PyMontbrioMeanField>()?;
    m.add_class::<PyBrunelNetwork>()?;
    m.add_class::<PyTUMNetwork>()?;
    m.add_class::<PyElBoustaniNetwork>()?;
    // misc
    m.add_class::<PyGradedSynapseNeuron>()?;
    m.add_class::<PyGapJunctionNeuron>()?;
    m.add_class::<PyFHAxon>()?;
    m.add_class::<PyNodeOfRanvier>()?;
    m.add_class::<PyMyelinatedAxon>()?;
    m.add_class::<PyCardiacPurkinjeFibre>()?;
    m.add_class::<PySmoothMuscleCell>()?;
    m.add_class::<PyEndocrineBetaCell>()?;
    // gap synapse models
    m.add_class::<PyTripletStdpSynapse>()?;
    m.add_class::<PyShortTermPlasticitySynapse>()?;
    m.add_class::<PyDopamineStdpSynapse>()?;
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
