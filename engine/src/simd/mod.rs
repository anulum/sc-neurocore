// SPDX-License-Identifier: AGPL-3.0-or-later
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li

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

    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa & wb).count_ones() as u64)
        .sum()
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

    a.iter()
        .zip(b.iter())
        .map(|(&wa, &wb)| (wa ^ wb).count_ones() as u64)
        .sum()
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
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_f64_neon(a, b) };
    }

    #[allow(unreachable_code)]
    {
        let len = a.len().min(b.len());
        a[..len].iter().zip(&b[..len]).map(|(&x, &y)| x * y).sum()
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
    a.iter().copied().fold(f64::NEG_INFINITY, f64::max)
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
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::sum_f64_neon(a) };
    }

    #[allow(unreachable_code)]
    a.iter().sum()
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
