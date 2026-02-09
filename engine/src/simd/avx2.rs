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

#[cfg(not(target_arch = "x86_64"))]
/// Fallback popcount when AVX2 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX2 variant.
pub unsafe fn popcount_avx2(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}
