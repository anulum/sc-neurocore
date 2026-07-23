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

    assert_eq!(models.len(), 162);
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
