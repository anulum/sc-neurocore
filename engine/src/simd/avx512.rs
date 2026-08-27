// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AVX512

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
    let (chunks, remainder) = data.as_chunks::<16>();

    for chunk in chunks {
        let v0 = _mm512_loadu_si512(chunk.as_ptr() as *const __m512i);
        let v1 = _mm512_loadu_si512(chunk.as_ptr().add(8) as *const __m512i);

        total += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(v0)) as u64;
        total += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(v1)) as u64;
    }

    total + crate::bitstream::popcount_words_portable(remainder)
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

    let (chunks, _) = data[..full_words].as_chunks_mut::<4>();
    let mut word_idx = 0;
    for chunk in chunks {
        let base = word_idx * 64;
        for i in 0..4 {
            let v = _mm512_loadu_si512(bits.as_ptr().add(base + i * 64) as *const __m512i);
            chunk[i] = _mm512_cmpneq_epi8_mask(v, zero);
        }
        word_idx += 4;
    }

    for i in word_idx..full_words {
        let v = _mm512_loadu_si512(bits.as_ptr().add(i * 64) as *const __m512i);
        data[i] = _mm512_cmpneq_epi8_mask(v, zero);
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
    let mut total = 0_u64;
    let (chunks_a, remainder_a) = a[..len].as_chunks::<16>();
    let (chunks_b, remainder_b) = b[..len].as_chunks::<16>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va0 = _mm512_loadu_si512(ca.as_ptr() as *const __m512i);
        let vb0 = _mm512_loadu_si512(cb.as_ptr() as *const __m512i);
        let va1 = _mm512_loadu_si512(ca.as_ptr().add(8) as *const __m512i);
        let vb1 = _mm512_loadu_si512(cb.as_ptr().add(8) as *const __m512i);

        let and0 = _mm512_and_si512(va0, vb0);
        let and1 = _mm512_and_si512(va1, vb1);

        total += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(and0)) as u64;
        total += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(and1)) as u64;
    }

    total
        + remainder_a
            .iter()
            .zip(remainder_b)
            .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
            .sum::<u64>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
/// Fused XOR+popcount over packed words using AVX-512 VPOPCNTDQ.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f` and `avx512vpopcntdq`.
pub unsafe fn fused_xor_popcount_avx512(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    let mut total = 0_u64;
    let (chunks_a, remainder_a) = a[..len].as_chunks::<16>();
    let (chunks_b, remainder_b) = b[..len].as_chunks::<16>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va0 = _mm512_loadu_si512(ca.as_ptr() as *const __m512i);
        let vb0 = _mm512_loadu_si512(cb.as_ptr() as *const __m512i);
        let va1 = _mm512_loadu_si512(ca.as_ptr().add(8) as *const __m512i);
        let vb1 = _mm512_loadu_si512(cb.as_ptr().add(8) as *const __m512i);

        let xor0 = _mm512_xor_si512(va0, vb0);
        let xor1 = _mm512_xor_si512(va1, vb1);

        total += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(xor0)) as u64;
        total += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(xor1)) as u64;
    }

    total
        + remainder_a
            .iter()
            .zip(remainder_b)
            .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
            .sum::<u64>()
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
    assert!(buf.len() >= 64, "buffer must contain at least 64 bytes");
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

// --- f64 SIMD operations (AVX-512: 8-wide f64) ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// Dot product of two f64 slices using AVX-512.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f`.
pub unsafe fn dot_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut acc = _mm512_setzero_pd();
    let (chunks_a, remainder_a) = a[..len].as_chunks::<8>();
    let (chunks_b, remainder_b) = b[..len].as_chunks::<8>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va = _mm512_loadu_pd(ca.as_ptr());
        let vb = _mm512_loadu_pd(cb.as_ptr());
        acc = _mm512_fmadd_pd(va, vb, acc);
    }

    let mut sum = _mm512_reduce_add_pd(acc);
    for (&ra, &rb) in remainder_a.iter().zip(remainder_b) {
        sum += ra * rb;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// Maximum of f64 slice using AVX-512.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f`.
pub unsafe fn max_f64_avx512(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut vmax0 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut vmax1 = _mm512_set1_pd(f64::NEG_INFINITY);
    let (chunks, remainder) = a.as_chunks::<16>();

    for chunk in chunks {
        vmax0 = _mm512_max_pd(vmax0, _mm512_loadu_pd(chunk.as_ptr()));
        vmax1 = _mm512_max_pd(vmax1, _mm512_loadu_pd(chunk.as_ptr().add(8)));
    }

    let mut m = _mm512_reduce_max_pd(_mm512_max_pd(vmax0, vmax1));
    for &v in remainder {
        m = m.max(v);
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// Sum of f64 slice using AVX-512.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f`.
pub unsafe fn sum_f64_avx512(a: &[f64]) -> f64 {
    let mut acc0 = _mm512_setzero_pd();
    let mut acc1 = _mm512_setzero_pd();
    let (chunks, remainder) = a.as_chunks::<16>();

    for chunk in chunks {
        acc0 = _mm512_add_pd(acc0, _mm512_loadu_pd(chunk.as_ptr()));
        acc1 = _mm512_add_pd(acc1, _mm512_loadu_pd(chunk.as_ptr().add(8)));
    }

    let mut sum = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));
    for &v in remainder {
        sum += v;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
/// Scale f64 slice in-place: y[i] *= alpha, using AVX-512.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx512f`.
pub unsafe fn scale_f64_avx512(alpha: f64, y: &mut [f64]) {
    let valpha = _mm512_set1_pd(alpha);
    let (chunks, remainder) = y.as_chunks_mut::<16>();

    for chunk in chunks {
        let v0 = _mm512_loadu_pd(chunk.as_ptr());
        let v1 = _mm512_loadu_pd(chunk.as_ptr().add(8));
        _mm512_storeu_pd(chunk.as_mut_ptr(), _mm512_mul_pd(v0, valpha));
        _mm512_storeu_pd(chunk.as_mut_ptr().add(8), _mm512_mul_pd(v1, valpha));
    }

    for v in remainder {
        *v *= alpha;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    a[..len].iter().zip(&b[..len]).map(|(&x, &y)| x * y).sum()
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn max_f64_avx512(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn sum_f64_avx512(a: &[f64]) -> f64 {
    a.iter().sum()
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn scale_f64_avx512(alpha: f64, y: &mut [f64]) {
    for v in y.iter_mut() {
        *v *= alpha;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
/// Compare 1024 random bytes against a threshold and return 16 u64 words.
///
/// # Safety
/// Caller must ensure AVX-512 BW is available on the current CPU.
pub unsafe fn bernoulli_compare_batch_avx512(buf: &[u8], threshold: u8, out: &mut [u64]) {
    let v_thresh = _mm512_set1_epi8(threshold as i8);
    for i in 0..16 {
        let chunk = &buf[i * 64..(i + 1) * 64];
        let v = _mm512_loadu_si512(chunk.as_ptr() as *const _);
        // AVX-512 has direct unsigned comparison
        out[i] = _mm512_cmplt_epu8_mask(v, v_thresh);
    }
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
