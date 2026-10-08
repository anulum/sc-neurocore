// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Caller-owned bitstream packing contracts

//! Packing preserves byte normalization and refuses output shapes before mutation.

use sc_neurocore_engine::{bitstream, simd};

#[test]
fn caller_owned_packing_matches_existing_public_packing() {
    for length in [
        0, 1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1031,
    ] {
        let storage: Vec<u8> = (0..length + 1)
            .map(|index| [0, 1, 2, 127, 128, 255][index % 6])
            .collect();
        // Offset input exercises unaligned SIMD loads and a partial final word.
        let bits = &storage[1..];
        let expected = bitstream::pack(bits);
        let mut portable = vec![u64::MAX; expected.data.len()];
        let mut dispatched = portable.clone();
        bitstream::pack_fast_into(bits, &mut portable).unwrap();
        simd::pack_dispatch_into(bits, &mut dispatched).unwrap();
        assert_eq!(portable, expected.data, "portable length={length}");
        assert_eq!(dispatched, expected.data, "dispatch length={length}");
        assert_eq!(bitstream::pack_fast(bits).data, expected.data);
        let legacy = simd::pack_dispatch(bits);
        assert_eq!(legacy.length, length);
        assert_eq!(legacy.data, expected.data);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let mut output = vec![u64::MAX; expected.data.len()];
                // SAFETY: Runtime detection proves AVX2; the borrowed slices
                // are valid and output is exclusively owned for the call.
                unsafe { simd::avx2::pack_avx2_into(bits, &mut output) }.unwrap();
                assert_eq!(output, expected.data, "AVX2 length={length}");
                assert_eq!(unsafe { simd::avx2::pack_avx2(bits) }, expected.data);
            }
            if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                let mut output = vec![u64::MAX; expected.data.len()];
                // SAFETY: Both AVX-512 target features were detected; input
                // and the exclusively borrowed output retain slice validity.
                unsafe { simd::avx512::pack_avx512_into(bits, &mut output) }.unwrap();
                assert_eq!(output, expected.data, "AVX-512 length={length}");
                assert_eq!(unsafe { simd::avx512::pack_avx512(bits) }, expected.data);
            }
        }
    }
}

#[test]
fn refused_output_shapes_preserve_every_destination_word() {
    let message = "packed output length must equal ceil(input length / 64)";
    for bits in [&[][..], &[255][..], &[255; 65][..]] {
        let words = bits.len().div_ceil(64);
        for size in [0, 1, 2, 3] {
            if size == words {
                continue;
            }
            let original = vec![0xACE1_u64; size];
            let mut output = original.clone();
            assert_eq!(bitstream::pack_fast_into(bits, &mut output), Err(message));
            assert_eq!(output, original);
            assert_eq!(simd::pack_dispatch_into(bits, &mut output), Err(message));
            assert_eq!(output, original);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    // SAFETY: Runtime detection proves AVX2 and slice validity
                    // holds even for a refused destination length.
                    assert_eq!(
                        unsafe { simd::avx2::pack_avx2_into(bits, &mut output) },
                        Err(message)
                    );
                    assert_eq!(output, original);
                }
                if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
                    // SAFETY: Both required CPU features are detected; length
                    // refusal must precede all destination mutation.
                    assert_eq!(
                        unsafe { simd::avx512::pack_avx512_into(bits, &mut output) },
                        Err(message)
                    );
                    assert_eq!(output, original);
                }
            }
        }
    }
}
