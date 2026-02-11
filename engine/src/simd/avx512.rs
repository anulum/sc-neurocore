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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
/// Fused AND+popcount over packed words using AVX-512 VPOPCNTDQ.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f` and `avx512vpopcntdq`.
pub unsafe fn fused_and_popcount_avx512(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    let mut total = _mm512_setzero_si512();
    let mut chunks_a = a[..len].chunks_exact(8);
    let mut chunks_b = b[..len].chunks_exact(8);

    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = _mm512_loadu_si512(ca.as_ptr() as *const __m512i);
        let vb = _mm512_loadu_si512(cb.as_ptr() as *const __m512i);
        let anded = _mm512_and_epi64(va, vb);
        let counts = _mm512_popcnt_epi64(anded);
        total = _mm512_add_epi64(total, counts);
    }

    let mut lanes = [0_u64; 8];
    _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, total);
    let mut sum: u64 = lanes.iter().sum();

    for (&wa, &wb) in chunks_a.remainder().iter().zip(chunks_b.remainder().iter()) {
        sum += (wa & wb).count_ones() as u64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
/// Fused XOR+popcount over packed words using AVX-512 VPOPCNTDQ.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f` and `avx512vpopcntdq`.
pub unsafe fn fused_xor_popcount_avx512(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    let mut total = _mm512_setzero_si512();
    let mut chunks_a = a[..len].chunks_exact(8);
    let mut chunks_b = b[..len].chunks_exact(8);

    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = _mm512_loadu_si512(ca.as_ptr() as *const __m512i);
        let vb = _mm512_loadu_si512(cb.as_ptr() as *const __m512i);
        let xored = _mm512_xor_epi64(va, vb);
        let counts = _mm512_popcnt_epi64(xored);
        total = _mm512_add_epi64(total, counts);
    }

    let mut lanes = [0_u64; 8];
    _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, total);
    let mut sum: u64 = lanes.iter().sum();

    for (&wa, &wb) in chunks_a.remainder().iter().zip(chunks_b.remainder().iter()) {
        sum += (wa ^ wb).count_ones() as u64;
    }
    sum
}

#[cfg(not(target_arch = "x86_64"))]
/// Fallback fused XOR+popcount when AVX-512 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX-512 variant.
pub unsafe fn fused_xor_popcount_avx512(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
/// Compare 64 random bytes against an unsigned threshold and return bit mask.
///
/// Bit `i` in the returned mask is 1 iff `buf[i] < threshold`.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f` and `avx512bw`.
/// `buf` must have at least 64 elements.
pub unsafe fn bernoulli_compare_avx512(buf: &[u8], threshold: u8) -> u64 {
    debug_assert!(buf.len() >= 64, "buffer must contain at least 64 bytes");
    let data = _mm512_loadu_si512(buf.as_ptr() as *const __m512i);
    let thresh = _mm512_set1_epi8(threshold as i8);
    _mm512_cmplt_epu8_mask(data, thresh)
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

#[cfg(not(target_arch = "x86_64"))]
/// Fallback fused AND+popcount when AVX-512 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX-512 variant.
pub unsafe fn fused_and_popcount_avx512(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
}

#[cfg(not(target_arch = "x86_64"))]
/// Fallback Bernoulli compare when AVX-512 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX-512 variant.
pub unsafe fn bernoulli_compare_avx512(buf: &[u8], threshold: u8) -> u64 {
    let mut mask = 0_u64;
    for (bit, &rb) in buf.iter().take(64).enumerate() {
        if rb < threshold {
            mask |= 1_u64 << bit;
        }
    }
    mask
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

    #[test]
    fn fused_and_popcount_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512vpopcntdq") {
            return;
        }

        let lengths = [1_usize, 7, 8, 15, 16, 17, 31, 32, 64, 128];
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
                .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
                .sum();

            // SAFETY: Runtime-guarded by feature detection in this test.
            let got = unsafe { super::fused_and_popcount_avx512(&a, &b) };
            assert_eq!(got, expected, "Mismatch at len={len}");
        }
    }

    #[test]
    fn bernoulli_compare_avx512_matches_scalar() {
        if !is_x86_feature_detected!("avx512bw") {
            return;
        }

        let buf: Vec<u8> = (0..64).map(|i| (i * 41 + 23) as u8).collect();
        let thresholds = [0_u8, 1, 2, 17, 64, 127, 128, 200, 255];

        for threshold in thresholds {
            let expected = buf.iter().enumerate().fold(0_u64, |acc, (bit, &rb)| {
                acc | (u64::from(rb < threshold) << bit)
            });

            // SAFETY: Runtime-guarded by feature detection in this test.
            let got = unsafe { super::bernoulli_compare_avx512(&buf, threshold) };
            assert_eq!(
                got, expected,
                "Mismatch for threshold={threshold} buf={buf:?}"
            );
        }
    }
}
