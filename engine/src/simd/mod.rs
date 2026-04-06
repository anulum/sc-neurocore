// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — SIMD Popcount Dispatch

//! # SIMD Popcount Dispatch
//!
//! Runtime CPU-feature dispatch for packed-bit popcount kernels.
//! Supported backends: AVX-512, AVX2, ARM NEON, ARM SVE, RISC-V RVV.

use rand::Rng;

pub mod avx2;
pub mod avx512;
pub mod neon;
pub mod rvv;
pub mod sve;

/// Pack u8 bits into u64 words using the best available SIMD path.
pub fn pack_dispatch(bits: &[u8]) -> crate::bitstream::BitStreamTensor {
    let length = bits.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            // SAFETY: Guarded by runtime feature detection.
            let data = unsafe { avx512::pack_avx512(bits) };
            return crate::bitstream::BitStreamTensor { data, length };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Guarded by runtime feature detection.
            let data = unsafe { avx2::pack_avx2(bits) };
            return crate::bitstream::BitStreamTensor { data, length };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
    {
        // SAFETY: SVE target feature is compile-time guaranteed.
        let data = unsafe { sve::pack_sve(bits) };
        return crate::bitstream::BitStreamTensor { data, length };
    }

    crate::bitstream::pack_fast(bits)
}

/// Count set bits in packed `u64` words using the best available SIMD path.
pub fn popcount_dispatch(data: &[u64]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            // SAFETY: Guarded by runtime feature detection.
            return unsafe { avx512::popcount_avx512(data) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Guarded by runtime feature detection.
            return unsafe { avx2::popcount_avx2(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "sve")]
        {
            // SAFETY: SVE target feature is compile-time guaranteed.
            return unsafe { sve::popcount_sve(data) };
        }
        #[cfg(not(target_feature = "sve"))]
        {
            // SAFETY: NEON is baseline on aarch64 targets.
            return unsafe { neon::popcount_neon(data) };
        }
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        // SAFETY: RVV target feature is compile-time guaranteed.
        return unsafe { rvv::popcount_rvv(data) };
    }

    crate::bitstream::popcount_words_portable(data)
}

/// Fused AND+popcount dispatch using the best available SIMD path.
pub fn fused_and_popcount_dispatch(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            // SAFETY: Guarded by runtime feature detection.
            return unsafe { avx512::fused_and_popcount_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Guarded by runtime feature detection.
            return unsafe { avx2::fused_and_popcount_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "sve")]
        {
            return unsafe { sve::fused_and_popcount_sve(a, b) };
        }
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        return unsafe { rvv::fused_and_popcount_rvv(a, b) };
    }

    let mut total = 0_u64;
    let mut chunks_a = a.chunks_exact(4);
    let mut chunks_b = b.chunks_exact(4);
    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        total += (ca[0] & cb[0]).count_ones() as u64;
        total += (ca[1] & cb[1]).count_ones() as u64;
        total += (ca[2] & cb[2]).count_ones() as u64;
        total += (ca[3] & cb[3]).count_ones() as u64;
    }
    total += chunks_a.remainder().iter()
        .zip(chunks_b.remainder().iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum::<u64>();
    total
}

/// Fused XOR+popcount dispatch using the best available SIMD path.
pub fn fused_xor_popcount_dispatch(a: &[u64], b: &[u64]) -> u64 {
    let len = a.len().min(b.len());
    let a = &a[..len];
    let b = &b[..len];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            // SAFETY: Guarded by runtime feature detection.
            return unsafe { avx512::fused_xor_popcount_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: Guarded by runtime feature detection.
            return unsafe { avx2::fused_xor_popcount_avx2(a, b) };
        }
        if is_x86_feature_detected!("avx") {
            let mut total = 0_u64;
            let mut chunks_a = a.chunks_exact(16);
            let mut chunks_b = b.chunks_exact(16);
            for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
                for i in 0..16 {
                    total += (ca[i] ^ cb[i]).count_ones() as u64;
                }
            }
            total += chunks_a.remainder().iter()
                .zip(chunks_b.remainder().iter())
                .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
                .sum::<u64>();
            return total;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "sve")]
        {
            return unsafe { sve::fused_xor_popcount_sve(a, b) };
        }
    }

    #[cfg(all(target_arch = "riscv64", target_feature = "v"))]
    {
        return unsafe { rvv::fused_xor_popcount_rvv(a, b) };
    }

    let mut total = 0_u64;
    let mut chunks_a = a.chunks_exact(4);
    let mut chunks_b = b.chunks_exact(4);
    for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        total += (ca[0] ^ cb[0]).count_ones() as u64;
        total += (ca[1] ^ cb[1]).count_ones() as u64;
        total += (ca[2] ^ cb[2]).count_ones() as u64;
        total += (ca[3] ^ cb[3]).count_ones() as u64;
    }
    total += chunks_a.remainder().iter()
        .zip(chunks_b.remainder().iter())
        .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
        .sum::<u64>();
    total
}

// --- f64 dispatch functions ---

/// Dot product of two f64 slices using the best available SIMD path.
pub fn dot_f64_dispatch(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512::dot_f64_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::dot_f64_avx2(a, b) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { avx2::dot_f64_avx(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_f64_neon(a, b) };
    }

    #[allow(unreachable_code)]
    {
        let len = a.len().min(b.len());
        let a_slice = &a[..len];
        let b_slice = &b[..len];
        let mut sum = 0.0_f64;
        let mut chunks_a = a_slice.chunks_exact(4);
        let mut chunks_b = b_slice.chunks_exact(4);
        for (ca, cb) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
            sum += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3];
        }
        sum += chunks_a.remainder().iter().zip(chunks_b.remainder()).map(|(x, y)| x * y).sum::<f64>();
        sum
    }
}

/// Maximum of f64 slice using the best available SIMD path.
pub fn max_f64_dispatch(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512::max_f64_avx512(a) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::max_f64_avx2(a) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::max_f64_neon(a) };
    }

    #[allow(unreachable_code)]
    {
        let mut m = f64::NEG_INFINITY;
        let mut chunks = a.chunks_exact(4);
        for c in chunks.by_ref() {
            m = m.max(c[0].max(c[1]).max(c[2].max(c[3])));
        }
        for &v in chunks.remainder() {
            m = m.max(v);
        }
        m
    }
}

/// Sum of f64 slice using the best available SIMD path.
pub fn sum_f64_dispatch(a: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { avx512::sum_f64_avx512(a) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::sum_f64_avx2(a) };
        }
        if is_x86_feature_detected!("avx") {
            return unsafe { avx2::sum_f64_avx(a) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::sum_f64_neon(a) };
    }

    #[allow(unreachable_code)]
    {
        let mut sum = 0.0_f64;
        let mut chunks = a.chunks_exact(4);
        for c in chunks.by_ref() {
            sum += c[0] + c[1] + c[2] + c[3];
        }
        sum += chunks.remainder().iter().sum::<f64>();
        sum
    }
}

/// Scale f64 slice in-place: y[i] *= alpha, using the best available SIMD path.
pub fn scale_f64_dispatch(alpha: f64, y: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { avx512::scale_f64_avx512(alpha, y) };
            return;
        }
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::scale_f64_avx2(alpha, y) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::scale_f64_neon(alpha, y) };
        return;
    }

    #[allow(unreachable_code)]
    for v in y.iter_mut() {
        *v *= alpha;
    }
}

/// Hamming distance between two packed bitstream slices.
pub fn hamming_distance_dispatch(a: &[u64], b: &[u64]) -> u64 {
    fused_xor_popcount_dispatch(a, b)
}

/// In-place softmax over an f64 slice (numerically stable).
///
/// Computes: subtract max → exp → normalize by sum.
/// Uses SIMD dispatch for max-find, scaling, and sum reduction.
pub fn softmax_inplace_f64_dispatch(scores: &mut [f64]) {
    if scores.is_empty() {
        return;
    }
    let max_val = max_f64_dispatch(scores);
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
    }
    let exp_sum = sum_f64_dispatch(scores);
    if exp_sum > 0.0 {
        scale_f64_dispatch(1.0 / exp_sum, scores);
    }
}

/// Fused encode+AND+popcount dispatch.
///
/// Delegates to the scalar-control implementation in `bitstream`,
/// which already performs SIMD Bernoulli compare where available.
pub fn encode_and_popcount_dispatch<R: Rng + ?Sized>(
    weight_words: &[u64],
    prob: f64,
    length: usize,
    rng: &mut R,
) -> u64 {
    crate::bitstream::encode_and_popcount(weight_words, prob, length, rng)
}

/// Batch compare 1024 bytes against threshold using best SIMD.
pub fn bernoulli_compare_batch_1024(buf: &[u8], threshold: u8, out: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") {
            return unsafe { avx512::bernoulli_compare_batch_avx512(buf, threshold, out) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::bernoulli_compare_batch_avx2(buf, threshold, out) };
        }
    }
    
    
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 fallback (available on all x86_64)
        use core::arch::x86_64::*;
        unsafe {
            let v_thresh = _mm_set1_epi8(threshold as i8);
            let bias = _mm_set1_epi8(i8::MIN);
            let v_thresh_biased = _mm_xor_si128(v_thresh, bias);
            
            for i in 0..16 {
                let chunk = &buf[i*64..(i+1)*64];
                let mut word = 0_u64;
                for j in 0..4 {
                    let v = _mm_loadu_si128(chunk.as_ptr().add(j*16) as *const __m128i);
                    let v_biased = _mm_xor_si128(v, bias);
                    let m = _mm_cmpgt_epi8(v_thresh_biased, v_biased);
                    let mask = _mm_movemask_epi8(m) as u32;
                    word |= (mask as u64) << (j * 16);
                }
                out[i] = word;
            }
            return;
        }
    }

    // Generic fallback: 16 scalar calls
    for i in 0..16 {
        out[i] = crate::bitstream::simd_bernoulli_compare_exposed(&buf[i*64..(i+1)*64], threshold);
    }
}
