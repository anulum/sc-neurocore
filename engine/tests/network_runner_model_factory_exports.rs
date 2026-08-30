// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Network runner model-factory public contract

use std::collections::BTreeSet;

use sc_neurocore_engine::network_runner::{
    create_neuron, create_population, supported_models, NeuronVariant,
};

#[test]
fn public_model_catalogue_is_unique_and_fully_constructible() {
    let models = supported_models();
    let unique: BTreeSet<_> = models.iter().copied().collect();

    assert_eq!(models.len(), 180);
    assert_eq!(unique.len(), models.len());
    for name in models {
        assert!(
            create_neuron(name).is_ok(),
            "canonical model {name} must remain constructible"
        );
    }
}

#[test]
fn public_model_factory_preserves_alias_and_error_contracts() {
    assert!(matches!(
        create_neuron("AdExNeuron"),
        Ok(NeuronVariant::AdEx(_))
    ));
    assert!(matches!(
        create_neuron("NonResettingLIFNeuron"),
        Ok(NeuronVariant::NonResettingLIF(_))
    ));
    assert!(matches!(
        create_neuron("SCNonResettingAdaptiveLIFNeuron"),
        Ok(NeuronVariant::SCNonResettingAdaptiveLIF(_))
    ));
    assert!(matches!(
        create_neuron("SCSigmaDeltaAccumulatorNeuron"),
        Ok(NeuronVariant::SCSigmaDeltaAccumulator(_))
    ));
    assert!(matches!(
        create_neuron("SCNormalizedEnergyLIFNeuron"),
        Ok(NeuronVariant::SCNormalizedEnergyLIF(_))
    ));
    assert!(matches!(
        create_neuron("McKeanNeuron"),
        Ok(NeuronVariant::McKean(_))
    ));
    assert!(matches!(
        create_neuron("SCTriangularMcKeanNeuron"),
        Ok(NeuronVariant::SCTriangularMcKean(_))
    ));
    assert!(matches!(
        create_neuron("SCStochasticRateAdaptationNeuron"),
        Ok(NeuronVariant::SCStochasticRateAdaptation(_))
    ));
    assert!(matches!(
        create_neuron("SCResettingWilsonHRNeuron"),
        Ok(NeuronVariant::SCResettingWilsonHR(_))
    ));
    assert!(matches!(
        create_neuron("SCUpwardCrossingRulkovMapNeuron"),
        Ok(NeuronVariant::SCUpwardCrossingRulkovMap(_))
    ));
    assert!(matches!(
        create_neuron("SCClippedLogisticBurstingMapNeuron"),
        Ok(NeuronVariant::SCClippedLogisticBurstingMap(_))
    ));
    assert!(matches!(
        create_neuron("SCClippedRationalRecoveryMapNeuron"),
        Ok(NeuronVariant::SCClippedRationalRecoveryMap(_))
    ));
    assert!(matches!(
        create_neuron("SCScaledResetAdaptiveIFNeuron"),
        Ok(NeuronVariant::SCScaledResetAdaptiveIF(_))
    ));
    assert!(matches!(
        create_neuron("SCInclusivePerfectIntegratorNeuron"),
        Ok(NeuronVariant::PerfectIntegrator(_))
    ));
    assert_eq!(
        create_neuron("not-a-neuron").err().as_deref(),
        Some("Unsupported model: 'not-a-neuron'")
    );

    let empty = create_population("Izhikevich", 0)
        .expect("zero-sized populations must remain constructible");
    assert!(empty.is_empty());
    assert_eq!(
        create_population("not-a-neuron", 2).err().as_deref(),
        Some("Unsupported model: 'not-a-neuron'")
    );
}

#[test]
fn network_runner_preserves_source_and_sc_perfect_integrator_boundaries() {
    let mut source = create_neuron("PerfectIntegratorNeuron").expect("source model must construct");
    let mut retained = create_neuron("SCInclusivePerfectIntegratorNeuron")
        .expect("retained SC model must construct");

    assert_eq!(source.step(5.0), 0);
    assert_eq!(retained.step(5.0), 0);
    assert_eq!(source.step(5.0), 0);
    assert_eq!(retained.step(5.0), 1);
    assert_eq!(source.step(5.0), 1);
    assert_eq!(retained.step(5.0), 0);
}

#[test]
fn network_runner_preserves_source_and_sc_quadratic_if_boundaries() {
    let source = create_neuron("QuadraticIFNeuron").expect("source model must construct");
    let retained =
        create_neuron("SCSymmetricQuadraticIFNeuron").expect("retained SC model must construct");
    assert!((source.soma_voltage() + 1.0).abs() < 1.0e-12);
    assert!((retained.soma_voltage() + 1.0).abs() < 1.0e-12);
}

#[test]
fn network_runner_preserves_theta_source_event_contract() {
    let mut source = create_neuron("ThetaNeuron").expect("theta source model must construct");
    let events: i32 = (0..1_000).map(|_| source.step(1.0)).sum();
    assert_eq!(events, 3);
}

#[test]
fn network_runner_preserves_dpi_source_event_contract() {
    let mut source = create_neuron("DPINeuron").expect("DPI source model must construct");
    let events: i32 = (0..5_000).map(|_| source.step(5.0)).sum();
    assert_eq!(events, 13);
    assert!((source.soma_voltage() - 0.24977022450967534).abs() < 1.0e-15);
}

#[test]
fn network_runner_preserves_both_rulkov_event_contracts() {
    let mut source = create_neuron("RulkovMapNeuron").expect("source model must construct");
    let mut retained =
        create_neuron("SCUpwardCrossingRulkovMapNeuron").expect("retained SC model must construct");

    assert_eq!(source.step(2.0), 0);
    assert_eq!(retained.step(2.0), 1);
    assert_eq!(source.soma_voltage(), retained.soma_voltage());
    assert_eq!(source.step(2.0), 0);
    assert_eq!(retained.step(2.0), 0);
    assert_eq!(source.step(2.0), 1);
    assert_eq!(retained.step(2.0), 0);
    assert_eq!(source.soma_voltage(), retained.soma_voltage());
}
