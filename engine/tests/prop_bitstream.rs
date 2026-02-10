use proptest::prelude::*;
use sc_neurocore_engine::bitstream::{bitwise_and, pack, popcount_words_portable, unpack};
use sc_neurocore_engine::simd::popcount_dispatch;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn pack_unpack_roundtrip(bits in prop::collection::vec(0u8..=1, 1..=4096)) {
        let packed = pack(&bits);
        let unpacked = unpack(&packed);
        prop_assert_eq!(&unpacked[..bits.len()], &bits[..]);
    }

    #[test]
    fn popcount_equals_sum(bits in prop::collection::vec(0u8..=1, 1..=4096)) {
        let expected: u64 = bits.iter().map(|b| *b as u64).sum();
        let packed = pack(&bits);
        let portable = popcount_words_portable(&packed.data);
        let dispatch = popcount_dispatch(&packed.data);
        prop_assert_eq!(portable, expected);
        prop_assert_eq!(dispatch, expected);
    }

    #[test]
    fn and_popcount_leq_min(
        a_bits in prop::collection::vec(0u8..=1, 64..=1024),
    ) {
        let b_bits: Vec<u8> = a_bits.iter().map(|x| if *x == 1 { 0 } else { 1 }).collect();
        let a = pack(&a_bits);
        let b = pack(&b_bits);
        let result = bitwise_and(&a, &b);
        let count = popcount_words_portable(&result.data);
        prop_assert_eq!(count, 0);
    }

    #[test]
    fn and_self_equals_self(bits in prop::collection::vec(0u8..=1, 64..=1024)) {
        let packed = pack(&bits);
        let result = bitwise_and(&packed, &packed);
        let self_count = popcount_words_portable(&packed.data);
        let result_count = popcount_words_portable(&result.data);
        prop_assert_eq!(self_count, result_count);
    }
}
