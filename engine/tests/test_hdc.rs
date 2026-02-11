// CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
// Contact us: www.anulum.li  protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AFFERO GENERAL PUBLIC LICENSE v3
// Commercial Licensing: Available

//! Integration tests for HDC / VSA operations on BitStreamTensor.

use sc_neurocore_engine::bitstream::{pack, unpack, BitStreamTensor};

// ── XOR truth table ──────────────────────────────────────────────────

#[test]
fn xor_self_is_zero() {
    let a = pack(&[1, 0, 1, 1, 0, 0, 1, 1]);
    let result = a.xor(&a);
    assert_eq!(unpack(&result), vec![0; 8]);
}

#[test]
fn xor_with_zero_is_identity() {
    let a = pack(&[1, 0, 1, 1, 0, 0, 1, 1]);
    let zero = pack(&[0, 0, 0, 0, 0, 0, 0, 0]);
    let result = a.xor(&zero);
    assert_eq!(unpack(&result), vec![1, 0, 1, 1, 0, 0, 1, 1]);
}

#[test]
fn xor_correctness() {
    let a = pack(&[1, 0, 1, 0]);
    let b = pack(&[1, 1, 0, 0]);
    let result = a.xor(&b);
    assert_eq!(unpack(&result), vec![0, 1, 1, 0]);
}

#[test]
fn xor_inplace_matches_xor() {
    let a = pack(&[1, 0, 1, 1, 0, 0, 1, 1]);
    let b = pack(&[0, 1, 1, 0, 1, 0, 0, 1]);
    let expected = a.xor(&b);
    let mut a_mut = a.clone();
    a_mut.xor_inplace(&b);
    assert_eq!(a_mut.data, expected.data);
}

// ── Rotate ───────────────────────────────────────────────────────────

#[test]
fn rotate_zero_is_identity() {
    let bits = vec![1, 0, 1, 1, 0, 0, 1, 1];
    let mut t = pack(&bits);
    t.rotate_right(0);
    assert_eq!(unpack(&t), bits);
}

#[test]
fn rotate_full_length_is_identity() {
    let bits = vec![1, 0, 1, 1, 0, 0, 1, 1];
    let n = bits.len();
    let mut t = pack(&bits);
    t.rotate_right(n);
    assert_eq!(unpack(&t), bits);
}

#[test]
fn rotate_right_one() {
    let bits = vec![1, 0, 1, 0];
    let mut t = pack(&bits);
    t.rotate_right(1);
    // Last element wraps to front
    assert_eq!(unpack(&t), vec![0, 1, 0, 1]);
}

#[test]
fn rotate_right_cross_word_boundary() {
    // 128 bits = 2 words: test cross-word rotation
    let mut bits = vec![0u8; 128];
    bits[0] = 1; // single bit at position 0
    let mut t = pack(&bits);
    t.rotate_right(1);
    let result = unpack(&t);
    // rotate_right(1): element at index i moves to (i+1)%n
    // So bit at position 0 moves to position 1
    assert_eq!(result[0], 0);
    assert_eq!(result[1], 1, "bit 0 should have shifted to position 1");

    // Test wrap-around: bit at position 127 should wrap to position 0
    let mut bits2 = vec![0u8; 128];
    bits2[127] = 1;
    let mut t2 = pack(&bits2);
    t2.rotate_right(1);
    let result2 = unpack(&t2);
    assert_eq!(result2[0], 1, "bit 127 should have wrapped to position 0");
    assert_eq!(result2[127], 0);
}

// ── Hamming distance ────────────────────────────────────────────────

#[test]
fn hamming_identical_is_zero() {
    let a = pack(&[1, 0, 1, 1, 0, 0, 1, 1]);
    assert_eq!(a.hamming_distance(&a), 0.0);
}

#[test]
fn hamming_complement_is_one() {
    let a = pack(&[1, 0, 1, 0, 1, 0, 1, 0]);
    let b = pack(&[0, 1, 0, 1, 0, 1, 0, 1]);
    assert_eq!(a.hamming_distance(&b), 1.0);
}

#[test]
fn hamming_large_random_near_half() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use sc_neurocore_engine::bitstream::bernoulli_packed;

    let dim = 10_000;
    let mut rng1 = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(99);
    let a = BitStreamTensor::from_words(bernoulli_packed(0.5, dim, &mut rng1), dim);
    let b = BitStreamTensor::from_words(bernoulli_packed(0.5, dim, &mut rng2), dim);
    let hd = a.hamming_distance(&b);
    assert!(
        (hd - 0.5).abs() < 0.05,
        "Random 10k-bit vectors should have ~0.5 hamming distance, got {hd}"
    );
}

// ── Bundle (majority vote) ──────────────────────────────────────────

#[test]
fn bundle_majority_three() {
    let a = pack(&[1, 1, 0, 0]);
    let b = pack(&[1, 0, 1, 0]);
    let c = pack(&[1, 0, 0, 1]);
    let result = BitStreamTensor::bundle(&[&a, &b, &c]);
    // Majority of [1,1,1]=1, [1,0,0]=0, [0,1,0]=0, [0,0,1]=0
    assert_eq!(unpack(&result), vec![1, 0, 0, 0]);
}

#[test]
fn bundle_single_is_identity() {
    let a = pack(&[1, 0, 1, 0, 1, 1]);
    let result = BitStreamTensor::bundle(&[&a]);
    assert_eq!(unpack(&result), vec![1, 0, 1, 0, 1, 1]);
}

#[test]
fn bundle_preserves_consensus() {
    // All inputs agree on bit 0 = 1
    let a = pack(&[1, 0, 0, 0]);
    let b = pack(&[1, 1, 0, 0]);
    let c = pack(&[1, 0, 1, 0]);
    let d = pack(&[1, 0, 0, 1]);
    let e = pack(&[1, 1, 1, 0]);
    let result = BitStreamTensor::bundle(&[&a, &b, &c, &d, &e]);
    let bits = unpack(&result);
    assert_eq!(bits[0], 1, "Unanimous bit should survive bundling");
}

// ── SIMD fused XOR+popcount ─────────────────────────────────────────

#[test]
fn fused_xor_popcount_matches_scalar() {
    let lengths = [1_usize, 7, 8, 15, 16, 17, 31, 32, 64, 128, 256];
    for len in lengths {
        let a: Vec<u64> = (0..len)
            .map(|i| (i as u64).wrapping_mul(0xD6E8_FD9D_5A2B_1C47) ^ 0x1357_9BDF_2468_ACE0)
            .collect();
        let b: Vec<u64> = (0..len)
            .map(|i| (i as u64).wrapping_mul(0x94D0_49BB_1331_11EB) ^ 0xF0F0_0F0F_AAAA_5555)
            .collect();

        let expected: u64 = a
            .iter()
            .zip(b.iter())
            .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
            .sum();

        let got = sc_neurocore_engine::simd::fused_xor_popcount_dispatch(&a, &b);
        assert_eq!(got, expected, "Mismatch at len={len}");
    }
}
