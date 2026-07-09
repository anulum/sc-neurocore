// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

use sc_neurocore_engine::bitstream::{bitwise_and, pack, popcount, unpack};
use sc_neurocore_engine::encoder::{BitstreamEncoder, Lfsr16};
use sc_neurocore_engine::neuron::FixedPointLif;
use sc_neurocore_engine::simd::popcount_dispatch;

#[test]
fn bitstream_roundtrip_and_popcount() {
    let bits: Vec<u8> = (0..1024).map(|i| (i % 7 == 0) as u8).collect();
    let packed = pack(&bits);
    assert_eq!(bits, unpack(&packed));
    assert_eq!(popcount(&packed), popcount_dispatch(&packed.data));
}

#[test]
fn bitstream_and_behaviour() {
    let a = pack(&[1, 0, 1, 1, 0, 0, 1, 0]);
    let b = pack(&[1, 1, 0, 1, 0, 1, 1, 0]);
    let c = bitwise_and(&a, &b);
    assert_eq!(unpack(&c), vec![1, 0, 0, 1, 0, 0, 1, 0]);
}

#[test]
fn lfsr_has_no_zero_state_for_known_cycle_prefix() {
    let mut lfsr = Lfsr16::new(0xACE1);
    for _ in 0..4096 {
        assert_ne!(lfsr.step(), 0);
    }
}

#[test]
fn encoder_compares_before_advancing() {
    let mut enc = BitstreamEncoder::new(16, 0xACE1);
    // reg starts at 0xACE1; compare 0xACE1 < 0xACE1 is false → bit=0
    assert_eq!(enc.step(0xACE1), 0);
    // After step, LFSR has advanced; compare against a value above the new reg
    assert_eq!(enc.step(u16::MAX), 1);
}

#[test]
fn fixed_point_lif_smoke() {
    let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
    let mut spikes = 0i32;
    let mut spike_steps = Vec::new();
    for step in 0..128 {
        let (spike, _) = lif.step(20, 256, 128, 0);
        spikes += spike;
        if spike != 0 {
            spike_steps.push(step);
        }
    }
    assert!(spikes > 0, "neuron must fire with strong input");
    // After each spike, next 2 steps must be silent (refractory_period=2).
    for &s in &spike_steps {
        if s + 2 < 128 {
            // Step s+1 and s+2 should not appear in spike_steps.
            assert!(
                !spike_steps.contains(&(s + 1)),
                "step {} should be refractory",
                s + 1
            );
            assert!(
                !spike_steps.contains(&(s + 2)),
                "step {} should be refractory",
                s + 2
            );
        }
    }
}
