// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SpiNNaker LIF engine unit tests

use super::*;

#[test]
fn spinnaker_exact_flow_matches_the_python_model_bit_for_bit() {
    // Python's SpiNNakerLIFNeuron after 100 steps of 25.0 from its defaults: spikes at steps
    // 33 and 68, v = -50.57825400371075 and refrac_count = 0.
    let mut n = SpiNNakerLIFNeuron::new();
    let spikes: Vec<i32> = (0..100).map(|_| n.step(25.0)).collect();
    assert_eq!(spikes.iter().sum::<i32>(), 2);
    assert_eq!((spikes[32], spikes[67]), (1, 1));
    assert_eq!(n.v, -50.57825400371075);
    assert_eq!(n.refrac_count, 0.0);
}

#[test]
fn spinnaker_refractory_count_stops_at_zero() {
    // dt = 0.3 does not divide tau_refrac = 2: the count ends at 0, not at -0.1.
    let mut n = SpiNNakerLIFNeuron::new();
    n.dt = 0.3;
    n.refrac_count = 0.2;
    assert_eq!(n.step(0.0), 0);
    assert_eq!(n.refrac_count, 0.0);
}

#[test]
fn spinnaker_fires() {
    let mut n = SpiNNakerLIFNeuron::new();
    let t: i32 = (0..200).map(|_| n.step(30.0)).sum();
    assert!(t > 0);
}
#[test]
fn spinnaker_silent() {
    let mut n = SpiNNakerLIFNeuron::new();
    let t: i32 = (0..200).map(|_| n.step(0.0)).sum();
    assert_eq!(t, 0);
}
#[test]
fn spinnaker_reset() {
    let mut n = SpiNNakerLIFNeuron::new();
    for _ in 0..50 {
        n.step(30.0);
    }
    n.reset();
    assert!((n.v - n.v_rest).abs() < 1e-10);
}
#[test]
fn spinnaker_bounded() {
    let mut n = SpiNNakerLIFNeuron::new();
    for _ in 0..1000 {
        n.step(1e4);
    }
    assert!(n.v.is_finite());
}
#[test]
fn spinnaker_nan_no_panic() {
    SpiNNakerLIFNeuron::new().step(f64::NAN);
}
