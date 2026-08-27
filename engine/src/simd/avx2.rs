// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — AVX2

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
    let (chunks, remainder) = data.as_chunks::<16>();

    for chunk in chunks {
        let v0 = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        let v1 = _mm256_loadu_si256(chunk.as_ptr().add(4) as *const __m256i);
        let v2 = _mm256_loadu_si256(chunk.as_ptr().add(8) as *const __m256i);
        let v3 = _mm256_loadu_si256(chunk.as_ptr().add(12) as *const __m256i);

        let mut lanes = [0_u64; 16];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v0);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(4) as *mut __m256i, v1);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(8) as *mut __m256i, v2);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(12) as *mut __m256i, v3);

        for &w in lanes.iter() {
            total += w.count_ones() as u64;
        }
    }

    total
        + remainder
            .iter()
            .map(|&w| w.count_ones() as u64)
            .sum::<u64>()
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

    let (chunks, _) = data[..full_words].as_chunks_mut::<4>();
    let mut word_idx = 0;
    for chunk in chunks {
        let base = word_idx * 64;
        for i in 0..4 {
            let b = base + i * 64;
            let lo = _mm256_loadu_si256(bits.as_ptr().add(b) as *const __m256i);
            let hi = _mm256_loadu_si256(bits.as_ptr().add(b + 32) as *const __m256i);
            let lo_mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, zero)) as u32);
            let hi_mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, zero)) as u32);
            chunk[i] = ((hi_mask as u64) << 32) | (lo_mask as u64);
        }
        word_idx += 4;
    }

    for i in word_idx..full_words {
        let base = i * 64;
        let lo = _mm256_loadu_si256(bits.as_ptr().add(base) as *const __m256i);
        let hi = _mm256_loadu_si256(bits.as_ptr().add(base + 32) as *const __m256i);
        let lo_mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, zero)) as u32);
        let hi_mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, zero)) as u32);
        data[i] = ((hi_mask as u64) << 32) | (lo_mask as u64);
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
    let (chunks_a, remainder_a) = a[..len].as_chunks::<16>();
    let (chunks_b, remainder_b) = b[..len].as_chunks::<16>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va0 = _mm256_loadu_si256(ca.as_ptr() as *const __m256i);
        let vb0 = _mm256_loadu_si256(cb.as_ptr() as *const __m256i);
        let va1 = _mm256_loadu_si256(ca.as_ptr().add(4) as *const __m256i);
        let vb1 = _mm256_loadu_si256(cb.as_ptr().add(4) as *const __m256i);
        let va2 = _mm256_loadu_si256(ca.as_ptr().add(8) as *const __m256i);
        let vb2 = _mm256_loadu_si256(cb.as_ptr().add(8) as *const __m256i);
        let va3 = _mm256_loadu_si256(ca.as_ptr().add(12) as *const __m256i);
        let vb3 = _mm256_loadu_si256(cb.as_ptr().add(12) as *const __m256i);

        let and0 = _mm256_and_si256(va0, vb0);
        let and1 = _mm256_and_si256(va1, vb1);
        let and2 = _mm256_and_si256(va2, vb2);
        let and3 = _mm256_and_si256(va3, vb3);

        let mut lanes = [0_u64; 16];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, and0);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(4) as *mut __m256i, and1);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(8) as *mut __m256i, and2);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(12) as *mut __m256i, and3);

        for &w in lanes.iter() {
            total += w.count_ones() as u64;
        }
    }

    total
        + remainder_a
            .iter()
            .zip(remainder_b)
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
    let (chunks_a, remainder_a) = a[..len].as_chunks::<16>();
    let (chunks_b, remainder_b) = b[..len].as_chunks::<16>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va0 = _mm256_loadu_si256(ca.as_ptr() as *const __m256i);
        let vb0 = _mm256_loadu_si256(cb.as_ptr() as *const __m256i);
        let va1 = _mm256_loadu_si256(ca.as_ptr().add(4) as *const __m256i);
        let vb1 = _mm256_loadu_si256(cb.as_ptr().add(4) as *const __m256i);
        let va2 = _mm256_loadu_si256(ca.as_ptr().add(8) as *const __m256i);
        let vb2 = _mm256_loadu_si256(cb.as_ptr().add(8) as *const __m256i);
        let va3 = _mm256_loadu_si256(ca.as_ptr().add(12) as *const __m256i);
        let vb3 = _mm256_loadu_si256(cb.as_ptr().add(12) as *const __m256i);

        let xor0 = _mm256_xor_si256(va0, vb0);
        let xor1 = _mm256_xor_si256(va1, vb1);
        let xor2 = _mm256_xor_si256(va2, vb2);
        let xor3 = _mm256_xor_si256(va3, vb3);

        let mut lanes = [0_u64; 16];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, xor0);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(4) as *mut __m256i, xor1);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(8) as *mut __m256i, xor2);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(12) as *mut __m256i, xor3);

        for &w in lanes.iter() {
            total += w.count_ones() as u64;
        }
    }

    total
        + remainder_a
            .iter()
            .zip(remainder_b)
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
    let (chunks_a, remainder_a) = a[..len].as_chunks::<4>();
    let (chunks_b, remainder_b) = b[..len].as_chunks::<4>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va = _mm256_loadu_pd(ca.as_ptr());
        let vb = _mm256_loadu_pd(cb.as_ptr());
        acc = _mm256_fmadd_pd(va, vb, acc);
    }

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (&ra, &rb) in remainder_a.iter().zip(remainder_b) {
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
    let (chunks, remainder) = a.as_chunks::<4>();

    for chunk in chunks {
        let va = _mm256_loadu_pd(chunk.as_ptr());
        vmax = _mm256_max_pd(vmax, va);
    }

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), vmax);
    let mut m = lanes[0].max(lanes[1]).max(lanes[2].max(lanes[3]));
    for &v in remainder {
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
    let (chunks, remainder) = a.as_chunks::<4>();

    for chunk in chunks {
        let va = _mm256_loadu_pd(chunk.as_ptr());
        acc = _mm256_add_pd(acc, va);
    }

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for &v in remainder {
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
    let (chunks, remainder) = y.as_chunks_mut::<16>();

    for chunk in chunks {
        let v0 = _mm256_loadu_pd(chunk.as_ptr());
        let v1 = _mm256_loadu_pd(chunk.as_ptr().add(4));
        let v2 = _mm256_loadu_pd(chunk.as_ptr().add(8));
        let v3 = _mm256_loadu_pd(chunk.as_ptr().add(12));

        _mm256_storeu_pd(chunk.as_mut_ptr(), _mm256_mul_pd(v0, valpha));
        _mm256_storeu_pd(chunk.as_mut_ptr().add(4), _mm256_mul_pd(v1, valpha));
        _mm256_storeu_pd(chunk.as_mut_ptr().add(8), _mm256_mul_pd(v2, valpha));
        _mm256_storeu_pd(chunk.as_mut_ptr().add(12), _mm256_mul_pd(v3, valpha));
    }

    for v in remainder {
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
    // let v_max = _mm256_set1_pd(max_val);

    let (chunks, remainder) = scores.as_chunks_mut::<16>();
    for chunk in chunks {
        // We still have to use scalar exp() because AVX2 does not have it in core::arch
        // but we can unroll the subtractions and stores.
        for i in 0..16 {
            chunk[i] = (chunk[i] - max_val).exp();
        }
    }
    for s in remainder {
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
    // let v_max = _mm256_set1_pd(max_val);

    let (chunks, remainder) = scores.as_chunks_mut::<16>();
    for chunk in chunks {
        // We still have to use scalar exp() because AVX2 does not have it in core::arch
        // but we can unroll the subtractions and stores.
        for i in 0..16 {
            chunk[i] = (chunk[i] - max_val).exp();
        }
    }
    for s in remainder {
        *s = (*s - max_val).exp();
    }

    let exp_sum = sum_f64_avx2(scores);
    if exp_sum > 0.0 {
        scale_f64_avx2(1.0 / exp_sum, scores);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
/// Dot product of two f64 slices using AVX (no FMA).
///
/// # Safety
/// Caller must ensure the current CPU supports `avx`.
pub unsafe fn dot_f64_avx(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();

    let (chunks_a, remainder_a) = a[..len].as_chunks::<16>();
    let (chunks_b, remainder_b) = b[..len].as_chunks::<16>();

    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        let va0 = _mm256_loadu_pd(ca.as_ptr());
        let vb0 = _mm256_loadu_pd(cb.as_ptr());
        acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(va0, vb0));

        let va1 = _mm256_loadu_pd(ca.as_ptr().add(4));
        let vb1 = _mm256_loadu_pd(cb.as_ptr().add(4));
        acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(va1, vb1));

        let va2 = _mm256_loadu_pd(ca.as_ptr().add(8));
        let vb2 = _mm256_loadu_pd(cb.as_ptr().add(8));
        acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(va2, vb2));

        let va3 = _mm256_loadu_pd(ca.as_ptr().add(12));
        let vb3 = _mm256_loadu_pd(cb.as_ptr().add(12));
        acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(va3, vb3));
    }

    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc0);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (&ra, &rb) in remainder_a.iter().zip(remainder_b) {
        sum += ra * rb;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
/// Sum of f64 slice using AVX.
///
/// # Safety
/// Caller must ensure AVX is available on the current CPU.
pub unsafe fn sum_f64_avx(a: &[f64]) -> f64 {
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();

    let (chunks, remainder) = a.as_chunks::<16>();
    for chunk in chunks {
        acc0 = _mm256_add_pd(acc0, _mm256_loadu_pd(chunk.as_ptr()));
        acc1 = _mm256_add_pd(acc1, _mm256_loadu_pd(chunk.as_ptr().add(4)));
        acc2 = _mm256_add_pd(acc2, _mm256_loadu_pd(chunk.as_ptr().add(8)));
        acc3 = _mm256_add_pd(acc3, _mm256_loadu_pd(chunk.as_ptr().add(12)));
    }

    acc0 = _mm256_add_pd(acc0, acc1);
    acc2 = _mm256_add_pd(acc2, acc3);
    acc0 = _mm256_add_pd(acc0, acc2);

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), acc0);
    let mut sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for &v in remainder {
        sum += v;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// Compare 1024 random bytes against a threshold and return 16 u64 words.
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
pub unsafe fn bernoulli_compare_batch_avx2(buf: &[u8], threshold: u8, out: &mut [u64]) {
    let v_thresh = _mm256_set1_epi8(threshold as i8);
    // Note: epi8 comparison is signed. Using the xor 0x80 trick for unsigned.
    let bias = _mm256_set1_epi8(i8::MIN);
    let v_thresh_biased = _mm256_xor_si256(v_thresh, bias);

    for i in 0..16 {
        // Each loop iteration processes 64 bytes (2x 256-bit registers)
        let chunk = &buf[i * 64..(i + 1) * 64];
        let v0 = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
        let v1 = _mm256_loadu_si256(chunk.as_ptr().add(32) as *const __m256i);

        let v0_biased = _mm256_xor_si256(v0, bias);
        let v1_biased = _mm256_xor_si256(v1, bias);

        let m0 = _mm256_cmpgt_epi8(v_thresh_biased, v0_biased);
        let m1 = _mm256_cmpgt_epi8(v_thresh_biased, v1_biased);

        let mask0 = _mm256_movemask_epi8(m0) as u32;
        let mask1 = _mm256_movemask_epi8(m1) as u32;
        out[i] = (mask0 as u64) | ((mask1 as u64) << 32);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
/// Maximum of f64 slice using AVX (v1).
///
/// # Safety
/// Caller must ensure AVX is available on the current CPU.
pub unsafe fn max_f64_avx(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut max_vec0 = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut max_vec1 = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut max_vec2 = _mm256_set1_pd(f64::NEG_INFINITY);
    let mut max_vec3 = _mm256_set1_pd(f64::NEG_INFINITY);

    let (chunks, remainder) = a.as_chunks::<16>();
    for chunk in chunks {
        max_vec0 = _mm256_max_pd(max_vec0, _mm256_loadu_pd(chunk.as_ptr()));
        max_vec1 = _mm256_max_pd(max_vec1, _mm256_loadu_pd(chunk.as_ptr().add(4)));
        max_vec2 = _mm256_max_pd(max_vec2, _mm256_loadu_pd(chunk.as_ptr().add(8)));
        max_vec3 = _mm256_max_pd(max_vec3, _mm256_loadu_pd(chunk.as_ptr().add(12)));
    }

    max_vec0 = _mm256_max_pd(max_vec0, max_vec1);
    max_vec2 = _mm256_max_pd(max_vec2, max_vec3);
    max_vec0 = _mm256_max_pd(max_vec0, max_vec2);

    let mut lanes = [0.0_f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), max_vec0);
    let mut m = lanes[0].max(lanes[1]).max(lanes[2].max(lanes[3]));
    for &v in remainder {
        m = m.max(v);
    }
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
/// Scale f64 slice using AVX (v1).
///
/// # Safety
/// Caller must ensure AVX is available on the current CPU.
pub unsafe fn scale_f64_avx(alpha: f64, y: &mut [f64]) {
    let valpha = _mm256_set1_pd(alpha);
    let (chunks, remainder) = y.as_chunks_mut::<16>();

    for chunk in chunks {
        let v0 = _mm256_loadu_pd(chunk.as_ptr());
        let v1 = _mm256_loadu_pd(chunk.as_ptr().add(4));
        let v2 = _mm256_loadu_pd(chunk.as_ptr().add(8));
        let v3 = _mm256_loadu_pd(chunk.as_ptr().add(12));

        _mm256_storeu_pd(chunk.as_mut_ptr(), _mm256_mul_pd(v0, valpha));
        _mm256_storeu_pd(chunk.as_mut_ptr().add(4), _mm256_mul_pd(v1, valpha));
        _mm256_storeu_pd(chunk.as_mut_ptr().add(8), _mm256_mul_pd(v2, valpha));
        _mm256_storeu_pd(chunk.as_mut_ptr().add(12), _mm256_mul_pd(v3, valpha));
    }

    for v in remainder {
        *v *= alpha;
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

    #[test]
    fn dot_f64_avx_matches_scalar() {
        if !is_x86_feature_detected!("avx") {
            return;
        }
        let a: Vec<f64> = (0..67).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..67).map(|i| (i as f64 * 0.3) - 5.0).collect();
        let expected: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let got = unsafe { super::dot_f64_avx(&a, &b) };
        assert!(
            (got - expected).abs() < 1e-9,
            "dot_avx: got {got}, expected {expected}"
        );
    }
}
