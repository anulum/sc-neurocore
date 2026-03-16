// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Fused AND+popcount over packed words using AVX2 for the AND stage.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn fused_and_popcount_avx2(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    let mut total = 0_u64;
    let mut chunks_a = a[..len].chunks_exact(4);
    let mut chunks_b = b[..len].chunks_exact(4);

    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = _mm256_loadu_si256(ca.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(cb.as_ptr() as *const __m256i);
        let anded = _mm256_and_si256(va, vb);

        let mut lanes = [0_u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, anded);
        total += lanes.iter().map(|w| w.count_ones() as u64).sum::<u64>();
    }

    total
        + chunks_a
            .remainder()
            .iter()
            .zip(chunks_b.remainder().iter())
            .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
            .sum::<u64>()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Fused XOR+popcount over packed words using AVX2 for the XOR stage.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn fused_xor_popcount_avx2(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    let mut total = 0_u64;
    let mut chunks_a = a[..len].chunks_exact(4);
    let mut chunks_b = b[..len].chunks_exact(4);

    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = _mm256_loadu_si256(ca.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(cb.as_ptr() as *const __m256i);
        let xored = _mm256_xor_si256(va, vb);

        let mut lanes = [0_u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, xored);
        total += lanes.iter().map(|w| w.count_ones() as u64).sum::<u64>();
    }

    total
        + chunks_a
            .remainder()
            .iter()
            .zip(chunks_b.remainder().iter())
            .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
            .sum::<u64>()
}

#[cfg(not(target_arch = "x86_64"))]
/// Fallback fused XOR+popcount when AVX2 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX2 variant.
pub unsafe fn fused_xor_popcount_avx2(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Compare 32 random bytes against an unsigned threshold and return bit mask.
///
/// Bit `i` in the returned mask is 1 iff `buf[i] < threshold`.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
/// `buf` must have at least 32 elements.
pub unsafe fn bernoulli_compare_avx2(buf: &[u8], threshold: u8) -> u32 {
    assert!(buf.len() >= 32, "buffer must contain at least 32 bytes");

    let data = _mm256_loadu_si256(buf.as_ptr() as *const __m256i);
    let bias = _mm256_set1_epi8(i8::MIN);
    let data_biased = _mm256_xor_si256(data, bias);
    let thresh_biased = _mm256_set1_epi8((threshold ^ 0x80) as i8);
    let lt = _mm256_cmpgt_epi8(thresh_biased, data_biased);
    _mm256_movemask_epi8(lt) as u32
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

#[cfg(not(target_arch = "x86_64"))]
/// Fallback fused AND+popcount when AVX2 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX2 variant.
pub unsafe fn fused_and_popcount_avx2(a: &[u64], b: &[u64]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
}

#[cfg(not(target_arch = "x86_64"))]
/// Fallback Bernoulli compare when AVX2 is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the AVX2 variant.
pub unsafe fn bernoulli_compare_avx2(buf: &[u8], threshold: u8) -> u32 {
    let mut mask = 0_u32;
    for (bit, &rb) in buf.iter().take(32).enumerate() {
        if rb < threshold {
            mask |= 1_u32 << bit;
        }
    }
    mask
}

// --- f64 SIMD operations (AVX2: 4-wide f64) ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
/// Dot product of two f64 slices using AVX2 FMA.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2` and `fma`.
pub unsafe fn dot_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut acc = _mm256_setzero_pd();
    let mut chunks_a = a[..len].chunks_exact(4);
    let mut chunks_b = b[..len].chunks_exact(4);

    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = _mm256_loadu_pd(ca.as_ptr());
        let vb = _mm256_loadu_pd(cb.as_ptr());
        acc = _mm256_fmadd_pd(va, vb, acc);
    }

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (&ra, &rb) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        sum += ra * rb;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Maximum of f64 slice using AVX2.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn max_f64_avx2(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut vmax = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut chunks = a.chunks_exact(4);

    for chunk in chunks.by_ref() {
        let va = _mm256_loadu_pd(chunk.as_ptr());
        vmax = _mm256_max_pd(vmax, va);
    }

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), vmax);
    let mut m = lanes[0].max(lanes[1]).max(lanes[2].max(lanes[3]));
    for &v in chunks.remainder() {
        m = m.max(v);
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Sum of f64 slice using AVX2.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn sum_f64_avx2(a: &[f64]) -> f64 {
    let mut acc = _mm256_setzero_pd();
    let mut chunks = a.chunks_exact(4);

    for chunk in chunks.by_ref() {
        let va = _mm256_loadu_pd(chunk.as_ptr());
        acc = _mm256_add_pd(acc, va);
    }

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for &v in chunks.remainder() {
        sum += v;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Scale f64 slice in-place: y[i] *= alpha, using AVX2.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn scale_f64_avx2(alpha: f64, y: &mut [f64]) {
    let valpha = _mm256_set1_pd(alpha);
    let mut chunks = y.chunks_exact_mut(4);

    for chunk in chunks.by_ref() {
        let vy = _mm256_loadu_pd(chunk.as_ptr());
        let scaled = _mm256_mul_pd(vy, valpha);
        _mm256_storeu_pd(chunk.as_mut_ptr(), scaled);
    }

    for v in chunks.into_remainder() {
        *v *= alpha;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    a[..len].iter().zip(&b[..len]).map(|(&x, &y)| x * y).sum()
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn max_f64_avx2(a: &[f64]) -> f64 {
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn sum_f64_avx2(a: &[f64]) -> f64 {
    a.iter().sum()
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn scale_f64_avx2(alpha: f64, y: &mut [f64]) {
    for v in y.iter_mut() {
        *v *= alpha;
    }
}

/// Hamming distance between two packed bitstream slices using AVX2.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn hamming_distance_avx2(a: &[u64], b: &[u64]) -> u64 {
    fused_xor_popcount_avx2(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// In-place softmax using AVX2 for max, sum, and scale steps.
///
/// # Safety
/// Caller must ensure the current CPU supports `avx2`.
pub unsafe fn softmax_inplace_f64_avx2(scores: &mut [f64]) {
    if scores.is_empty() {
        return;
    }
    let max_val = max_f64_avx2(scores);
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
    }
    let exp_sum = sum_f64_avx2(scores);
    if exp_sum > 0.0 {
        scale_f64_avx2(1.0 / exp_sum, scores);
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn softmax_inplace_f64_avx2(scores: &mut [f64]) {
    if scores.is_empty() {
        return;
    }
    let max_val = max_f64_avx2(scores);
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
    }
    let exp_sum = sum_f64_avx2(scores);
    if exp_sum > 0.0 {
        scale_f64_avx2(1.0 / exp_sum, scores);
    }
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

    #[test]
    fn fused_and_popcount_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let lengths = [1_usize, 7, 8, 15, 16, 17, 31, 32, 64, 128];
        for len in lengths {
            let a: Vec<u64> = (0..len)
                .map(|i| (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5_A5A5_5A5A_5A5A)
                .collect();
            let b: Vec<u64> = (0..len)
                .map(|i| (i as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ 0x0F0F_F0F0_33CC_CC33)
                .collect();

            let expected: u64 = a
                .iter()
                .zip(b.iter())
                .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
                .sum();

            // SAFETY: Runtime-guarded by feature detection in this test.
            let got = unsafe { super::fused_and_popcount_avx2(&a, &b) };
            assert_eq!(got, expected, "Mismatch at len={len}");
        }
    }

    #[test]
    fn dot_f64_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let a: Vec<f64> = (0..67).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..67).map(|i| (i as f64 * 0.3) - 5.0).collect();
        let expected: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let got = unsafe { super::dot_f64_avx2(&a, &b) };
        assert!(
            (got - expected).abs() < 1e-9,
            "dot: got {got}, expected {expected}"
        );
    }

    #[test]
    fn max_f64_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a: Vec<f64> = (0..67).map(|i| (i as f64 * 7.3).sin()).collect();
        let expected = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let got = unsafe { super::max_f64_avx2(&a) };
        assert!(
            (got - expected).abs() < 1e-12,
            "max: got {got}, expected {expected}"
        );
    }

    #[test]
    fn sum_f64_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let a: Vec<f64> = (0..67).map(|i| i as f64 * 0.01).collect();
        let expected: f64 = a.iter().sum();
        let got = unsafe { super::sum_f64_avx2(&a) };
        assert!(
            (got - expected).abs() < 1e-9,
            "sum: got {got}, expected {expected}"
        );
    }

    #[test]
    fn softmax_avx2_sums_to_one() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut scores: Vec<f64> = (0..67).map(|i| (i as f64 * 0.3) - 10.0).collect();
        unsafe { super::softmax_inplace_f64_avx2(&mut scores) };
        let sum: f64 = scores.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "softmax must sum to 1.0, got {sum}"
        );
        assert!(scores.iter().all(|&s| s >= 0.0), "all values must be >= 0");
    }

    #[test]
    fn bernoulli_compare_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let buf: Vec<u8> = (0..32).map(|i| (i * 73 + 17) as u8).collect();
        let thresholds = [0_u8, 1, 2, 17, 64, 127, 128, 200, 255];

        for threshold in thresholds {
            let expected = buf.iter().enumerate().fold(0_u32, |acc, (bit, &rb)| {
                acc | (u32::from(rb < threshold) << bit)
            });

            // SAFETY: Runtime-guarded by feature detection in this test.
            let got = unsafe { super::bernoulli_compare_avx2(&buf, threshold) };
            assert_eq!(
                got, expected,
                "Mismatch for threshold={threshold} buf={buf:?}"
            );
        }
    }
}
