#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
/// Count set bits in 64-bit words using AVX-512 VPOPCNTDQ.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f` and `avx512vpopcntdq`.
pub unsafe fn popcount_avx512(data: &[u64]) -> u64 {
    let mut total = 0_u64;
    let mut chunks = data.chunks_exact(8);

    for chunk in &mut chunks {
        let v = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
        let counts = _mm512_popcnt_epi64(v);
        let mut lanes = [0_u64; 8];
        _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, counts);
        total += lanes.iter().sum::<u64>();
    }

    total + crate::bitstream::popcount_words_portable(chunks.remainder())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
/// Pack u8 bits into u64 words using AVX-512 k-mask compare.
///
/// Processes 64 bytes per iteration where each compare result bit maps
/// directly to one packed output bit.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f` and `avx512bw`.
pub unsafe fn pack_avx512(bits: &[u8]) -> Vec<u64> {
    let length = bits.len();
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];
    let full_words = length / 64;
    let zero = _mm512_setzero_si512();

    for (word_idx, word) in data.iter_mut().take(full_words).enumerate() {
        let base = word_idx * 64;
        let v = _mm512_loadu_si512(bits.as_ptr().add(base) as *const __m512i);
        let mask = _mm512_cmpneq_epi8_mask(v, zero);
        *word = mask;
    }

    if full_words < words {
        let tail_start = full_words * 64;
        let tail = crate::bitstream::pack_fast(&bits[tail_start..]);
        data[full_words] = tail.data.first().copied().unwrap_or(0);
    }

    data
}

#[cfg(not(target_arch = "x86_64"))]
/// Fallback popcount when AVX-512 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX-512 variant.
pub unsafe fn popcount_avx512(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}

#[cfg(not(target_arch = "x86_64"))]
/// Fallback pack when AVX-512 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX-512 variant.
pub unsafe fn pack_avx512(bits: &[u8]) -> Vec<u64> {
    crate::bitstream::pack_fast(bits).data
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use crate::bitstream::pack;

    #[test]
    fn pack_avx512_matches_pack() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        let lengths = [
            1_usize, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1024, 1031,
        ];
        for length in lengths {
            let bits: Vec<u8> = (0..length)
                .map(|i| if (i * 19 + 11) % 4 == 0 { 1 } else { 0 })
                .collect();
            // SAFETY: Runtime-guarded by feature detection in this test.
            let got = unsafe { super::pack_avx512(&bits) };
            let expected = pack(&bits).data;
            assert_eq!(got, expected, "Mismatch at length={length}");
        }
    }
}
