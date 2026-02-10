#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Count set bits in 64-bit words using AVX2.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn popcount_avx2(data: &[u64]) -> u64 {
    let mut total = 0_u64;
    let mut chunks = data.chunks_exact(4);

    let m1 = _mm256_set1_epi64x(0x5555_5555_5555_5555_u64 as i64);
    let m2 = _mm256_set1_epi64x(0x3333_3333_3333_3333_u64 as i64);
    let m4 = _mm256_set1_epi64x(0x0f0f_0f0f_0f0f_0f0f_u64 as i64);

    for chunk in &mut chunks {
        let mut x = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        x = _mm256_sub_epi64(x, _mm256_and_si256(_mm256_srli_epi64::<1>(x), m1));
        x = _mm256_add_epi64(
            _mm256_and_si256(x, m2),
            _mm256_and_si256(_mm256_srli_epi64::<2>(x), m2),
        );
        x = _mm256_and_si256(_mm256_add_epi64(x, _mm256_srli_epi64::<4>(x)), m4);

        let mut lanes = [0_u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, x);
        total += lanes
            .iter()
            .copied()
            .map(|lane| lane.wrapping_mul(0x0101_0101_0101_0101) >> 56)
            .sum::<u64>();
    }

    total + crate::bitstream::popcount_words_portable(chunks.remainder())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Pack u8 bits into u64 words using AVX2 movemask.
///
/// Processes 64 bytes into one u64 word by building two 32-bit masks.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn pack_avx2(bits: &[u8]) -> Vec<u64> {
    let length = bits.len();
    let words = length.div_ceil(64);
    let mut data = vec![0_u64; words];
    let full_words = length / 64;
    let zero = _mm256_setzero_si256();

    for (word_idx, word) in data.iter_mut().take(full_words).enumerate() {
        let base = word_idx * 64;
        let lo = _mm256_loadu_si256(bits.as_ptr().add(base) as *const __m256i);
        let hi = _mm256_loadu_si256(bits.as_ptr().add(base + 32) as *const __m256i);

        let lo_eq_zero = _mm256_cmpeq_epi8(lo, zero);
        let hi_eq_zero = _mm256_cmpeq_epi8(hi, zero);
        let lo_mask = !(_mm256_movemask_epi8(lo_eq_zero) as u32);
        let hi_mask = !(_mm256_movemask_epi8(hi_eq_zero) as u32);

        *word = ((hi_mask as u64) << 32) | (lo_mask as u64);
    }

    if full_words < words {
        let tail_start = full_words * 64;
        let tail = crate::bitstream::pack_fast(&bits[tail_start..]);
        data[full_words] = tail.data.first().copied().unwrap_or(0);
    }

    data
}

#[cfg(not(target_arch = "x86_64"))]
/// Fallback popcount when AVX2 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX2 variant.
pub unsafe fn popcount_avx2(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}

#[cfg(not(target_arch = "x86_64"))]
/// Fallback pack when AVX2 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX2 variant.
pub unsafe fn pack_avx2(bits: &[u8]) -> Vec<u64> {
    crate::bitstream::pack_fast(bits).data
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use crate::bitstream::pack;

    #[test]
    fn pack_avx2_matches_pack() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let lengths = [
            1_usize, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1024, 1031,
        ];
        for length in lengths {
            let bits: Vec<u8> = (0..length)
                .map(|i| if (i * 17 + 5) % 3 == 0 { 1 } else { 0 })
                .collect();
            // SAFETY: Runtime-guarded by feature detection in this test.
            let got = unsafe { super::pack_avx2(&bits) };
            let expected = pack(&bits).data;
            assert_eq!(got, expected, "Mismatch at length={length}");
        }
    }
}
