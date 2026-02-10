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
fn encoder_is_step_then_compare() {
    let mut enc = BitstreamEncoder::new(16, 0xACE1);
    let bit = enc.step(0xACE1);
    assert_eq!(bit, 1);
}

#[test]
fn fixed_point_lif_smoke() {
    let mut lif = FixedPointLif::new(16, 8, 0, 0, 256, 2);
    let mut spikes = 0;
    for _ in 0..128 {
        let (spike, _) = lif.step(20, 256, 128, 0);
        spikes += spike;
    }
    assert_eq!(spikes, 0);
}
