// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

use sc_neurocore_engine::neurons::rate::CompteWMNeuron;
use sc_neurocore_engine::sc_compte_wm_network::{
    counter_poisson_counts, SCCompteWMNetwork, SCCompteWMNetworkSpec, SCCompteWMNetworkState,
    N_EXCITATORY, N_INHIBITORY,
};

#[test]
fn counter_stream_matches_python_fixture() {
    let counts = counter_poisson_counts(64, 1800.0, 0.02, 42, 0, 0).unwrap();
    let active: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(i, n)| (*n > 0).then_some(i))
        .collect();
    assert_eq!(active, [49, 61]);
    assert_eq!(counts.iter().sum::<u64>(), 2);
}

#[test]
fn isolated_external_impulse_matches_preserved_cell() {
    let mut network = SCCompteWMNetwork::new(SCCompteWMNetworkSpec::default(), None).unwrap();
    let mut exc = vec![0; N_EXCITATORY];
    let inh = vec![0; N_INHIBITORY];
    exc[17] = 1;
    network
        .step_with_events(&vec![0.0; N_EXCITATORY], &exc, &inh)
        .unwrap();
    let mut cell = CompteWMNeuron::new();
    assert_eq!(cell.step_events(0.0, false, true, false), Ok(0));
    assert!((network.state.v_exc_mv[17] - cell.v).abs() < 2.0e-14);
    assert!((network.state.external_ampa_exc[17] - cell.s_ampa).abs() < 2.0e-14);
}

#[test]
fn invalid_input_is_atomic() {
    let mut network = SCCompteWMNetwork::new(SCCompteWMNetworkSpec::default(), None).unwrap();
    let before = network.state.clone();
    assert!(network.step_with_events(&[], &[], &[]).is_err());
    assert_eq!(network.state, before);
}

#[test]
fn fft_network_step_matches_python_dense_oracle_fixture() {
    let mut state = SCCompteWMNetworkState::default();
    state.v_exc_mv.fill(-60.0);
    state.recurrent_nmda[0] = 0.2;
    state.recurrent_nmda[37] = 0.4;
    state.recurrent_nmda[1024] = 0.1;
    state.recurrent_nmda[1901] = 0.3;
    let mut network =
        SCCompteWMNetwork::new(SCCompteWMNetworkSpec::default(), Some(state)).unwrap();
    network
        .step_with_events(
            &vec![0.0; N_EXCITATORY],
            &vec![0; N_EXCITATORY],
            &vec![0; N_INHIBITORY],
        )
        .unwrap();
    assert!((network.state.v_exc_mv[113] - (-60.009_906_823_044_3)).abs() < 3.0e-13);
    assert!((network.state.recurrent_nmda[37] - 0.399_920_008).abs() < 2.0e-15);
}
