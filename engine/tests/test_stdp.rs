// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

use sc_neurocore_engine::synapses::{StdpParams, StdpSynapse};

fn default_params() -> StdpParams {
    StdpParams {
        a_plus: 64,  // 0.25 in Q8.8
        a_minus: 48, // 0.1875 in Q8.8
        decay: 230,  // ~0.90 in Q8.8
        w_min: 0,
        w_max: 255,
    }
}

#[test]
fn ltp_pre_before_post_increases_weight() {
    let params = default_params();
    let mut syn = StdpSynapse::new(128, 16, 8);

    // Build pre-trace with several pre-spikes (no post)
    for _ in 0..5 {
        syn.step(true, false, &params);
    }
    let w_before = syn.weight;

    // Post-spike reads accumulated pre-trace → LTP
    syn.step(false, true, &params);
    assert!(syn.weight > w_before);
}

#[test]
fn ltd_post_before_pre_decreases_weight() {
    let params = default_params();
    let mut syn = StdpSynapse::new(200, 16, 8);

    // Build post-trace with several post-spikes (no pre)
    for _ in 0..5 {
        syn.step(false, true, &params);
    }
    let w_before = syn.weight;

    // Pre-spike reads accumulated post-trace → LTD
    syn.step(true, false, &params);
    assert!(syn.weight < w_before);
}

#[test]
fn simultaneous_spikes_trigger_ltp_only() {
    let params = default_params();
    let mut syn = StdpSynapse::new(128, 16, 8);

    // Build pre-trace first
    for _ in 0..5 {
        syn.step(true, false, &params);
    }
    let w_before = syn.weight;

    // Simultaneous: the if/else-if makes post_spike branch win (LTP)
    syn.step(true, true, &params);
    assert!(
        syn.weight >= w_before,
        "simultaneous spikes must not decrease weight (LTP path)"
    );
}

#[test]
fn weight_stays_within_bounds() {
    let params = default_params();

    // Hammer LTP from w_min
    let mut syn = StdpSynapse::new(params.w_min, 16, 8);
    for _ in 0..500 {
        syn.step(true, false, &params);
        syn.step(false, true, &params);
    }
    assert!(syn.weight >= params.w_min);
    assert!(syn.weight <= params.w_max);

    // Hammer LTD from w_max
    let mut syn = StdpSynapse::new(params.w_max, 16, 8);
    for _ in 0..500 {
        syn.step(false, true, &params);
        syn.step(true, false, &params);
    }
    assert!(syn.weight >= params.w_min);
    assert!(syn.weight <= params.w_max);
}

#[test]
fn positive_a_minus_does_not_potentiate_in_depression() {
    // a_minus is positive in this API (absolute magnitude for LTD).
    // Verify that the depression path still subtracts weight.
    let params = StdpParams {
        a_plus: 64,
        a_minus: 100, // large positive value
        decay: 230,
        w_min: 0,
        w_max: 255,
    };
    let mut syn = StdpSynapse::new(200, 16, 8);

    // Build post-trace
    for _ in 0..5 {
        syn.step(false, true, &params);
    }
    let w_before = syn.weight;

    // Pre-spike triggers depression path: weight -= |trace_post * a_minus| >> frac
    syn.step(true, false, &params);
    assert!(
        syn.weight <= w_before,
        "depression path must not increase weight even with positive a_minus"
    );
}
