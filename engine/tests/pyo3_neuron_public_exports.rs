// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — PyO3 neuron public-export contract

use std::any::TypeId;
use std::collections::HashSet;

use sc_neurocore_engine::pyo3_neurons::{
    PyAdaptiveThresholdIFNeuron, PyAiharaMapNeuron, PyAstrocyteLIFNeuron, PyAvRonCardiacNeuron,
    PyBalancedResonateAndFireNeuron, PyBertramPhantomBurster, PyButeraRespiratoryNeuron,
    PyCazellesMapNeuron, PyChayKeizerNeuron, PyChayNeuron, PyChialvoMapNeuron,
    PyClosedFormContinuousNeuron, PyComplementaryLIFNeuron, PyConnorStevensNeuron,
    PyCourageNekorkinMapNeuron, PyDeSchutterPurkinjeNeuron, PyDendriticNMDANeuron,
    PyDestexheThalamicNeuron, PyDurstewitzDopamineNeuron, PyEnergyLIFNeuron,
    PyErmentroutKopellMapNeuron, PyFitzHughNagumoNeuron, PyFitzHughRinzelNeuron,
    PyGIFPopulationNeuron, PyGLIFNeuron, PyGatedLIFNeuron, PyGolombFSNeuron,
    PyGutkinErmentroutNeuron, PyHillTononiNeuron, PyHindmarshRoseNeuron, PyHodgkinHuxleyNeuron,
    PyHuberBraunNeuron, PyIbarzTanakaMapNeuron, PyInhibitoryLIFNeuron, PyKLIFNeuron,
    PyKilincBhattMapNeuron, PyLearnableNeuronModel, PyMATNeuron, PyMainenSejnowskiNeuron,
    PyMcKeanNeuron, PyMedvedevMapNeuron, PyMihalasNieburNeuron, PyMorrisLecarNeuron,
    PyMulticompartmentMCNNeuron, PyNonResettingLIFNeuron, PyNonlinearLIFNeuron,
    PyParametricLIFNeuron, PyPerfectIntegratorNeuron, PyPernarowskiNeuron, PyPlantR15Neuron,
    PyPospischilNeuron, PyPrescottNeuron, PyQuadraticIFNeuron, PyResonateAndFireNeuron,
    PyRulkovMapNeuron, PySFANeuron, PyShermanRinzelKeizerNeuron, PySigmaDeltaNeuron,
    PyTermanWangOscillator, PyThetaNeuron, PyTraubMilesNeuron, PyWangBuzsakiNeuron,
    PyWilsonHRNeuron, PyYamadaNeuron,
};

#[test]
fn decomposed_wrappers_remain_distinct_public_types() {
    let exports = [
        TypeId::of::<PyAdaptiveThresholdIFNeuron>(),
        TypeId::of::<PyAiharaMapNeuron>(),
        TypeId::of::<PyAstrocyteLIFNeuron>(),
        TypeId::of::<PyAvRonCardiacNeuron>(),
        TypeId::of::<PyBalancedResonateAndFireNeuron>(),
        TypeId::of::<PyBertramPhantomBurster>(),
        TypeId::of::<PyButeraRespiratoryNeuron>(),
        TypeId::of::<PyCazellesMapNeuron>(),
        TypeId::of::<PyChayKeizerNeuron>(),
        TypeId::of::<PyChayNeuron>(),
        TypeId::of::<PyChialvoMapNeuron>(),
        TypeId::of::<PyClosedFormContinuousNeuron>(),
        TypeId::of::<PyComplementaryLIFNeuron>(),
        TypeId::of::<PyConnorStevensNeuron>(),
        TypeId::of::<PyCourageNekorkinMapNeuron>(),
        TypeId::of::<PyDeSchutterPurkinjeNeuron>(),
        TypeId::of::<PyDendriticNMDANeuron>(),
        TypeId::of::<PyDestexheThalamicNeuron>(),
        TypeId::of::<PyDurstewitzDopamineNeuron>(),
        TypeId::of::<PyEnergyLIFNeuron>(),
        TypeId::of::<PyErmentroutKopellMapNeuron>(),
        TypeId::of::<PyFitzHughNagumoNeuron>(),
        TypeId::of::<PyFitzHughRinzelNeuron>(),
        TypeId::of::<PyGIFPopulationNeuron>(),
        TypeId::of::<PyGLIFNeuron>(),
        TypeId::of::<PyGatedLIFNeuron>(),
        TypeId::of::<PyGolombFSNeuron>(),
        TypeId::of::<PyGutkinErmentroutNeuron>(),
        TypeId::of::<PyHillTononiNeuron>(),
        TypeId::of::<PyHindmarshRoseNeuron>(),
        TypeId::of::<PyHodgkinHuxleyNeuron>(),
        TypeId::of::<PyHuberBraunNeuron>(),
        TypeId::of::<PyIbarzTanakaMapNeuron>(),
        TypeId::of::<PyInhibitoryLIFNeuron>(),
        TypeId::of::<PyKLIFNeuron>(),
        TypeId::of::<PyKilincBhattMapNeuron>(),
        TypeId::of::<PyLearnableNeuronModel>(),
        TypeId::of::<PyMATNeuron>(),
        TypeId::of::<PyMainenSejnowskiNeuron>(),
        TypeId::of::<PyMcKeanNeuron>(),
        TypeId::of::<PyMedvedevMapNeuron>(),
        TypeId::of::<PyMihalasNieburNeuron>(),
        TypeId::of::<PyMorrisLecarNeuron>(),
        TypeId::of::<PyMulticompartmentMCNNeuron>(),
        TypeId::of::<PyNonResettingLIFNeuron>(),
        TypeId::of::<PyNonlinearLIFNeuron>(),
        TypeId::of::<PyParametricLIFNeuron>(),
        TypeId::of::<PyPerfectIntegratorNeuron>(),
        TypeId::of::<PyPernarowskiNeuron>(),
        TypeId::of::<PyPlantR15Neuron>(),
        TypeId::of::<PyPospischilNeuron>(),
        TypeId::of::<PyPrescottNeuron>(),
        TypeId::of::<PyQuadraticIFNeuron>(),
        TypeId::of::<PyResonateAndFireNeuron>(),
        TypeId::of::<PyRulkovMapNeuron>(),
        TypeId::of::<PySFANeuron>(),
        TypeId::of::<PyShermanRinzelKeizerNeuron>(),
        TypeId::of::<PySigmaDeltaNeuron>(),
        TypeId::of::<PyTermanWangOscillator>(),
        TypeId::of::<PyThetaNeuron>(),
        TypeId::of::<PyTraubMilesNeuron>(),
        TypeId::of::<PyWangBuzsakiNeuron>(),
        TypeId::of::<PyWilsonHRNeuron>(),
        TypeId::of::<PyYamadaNeuron>(),
    ];
    let unique: HashSet<_> = exports.into_iter().collect();

    assert_eq!(unique.len(), exports.len());
}
