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

#[cfg(not(target_arch = "x86_64"))]
/// Fallback popcount when AVX-512 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX-512 variant.
pub unsafe fn popcount_avx512(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}
