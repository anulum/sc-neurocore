//! # SIMD Popcount Dispatch
//!
//! Runtime CPU-feature dispatch for packed-bit popcount kernels.

pub mod avx2;
pub mod avx512;
pub mod neon;

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
        // SAFETY: NEON is baseline on aarch64 targets.
        return unsafe { neon::popcount_neon(data) };
    }

    crate::bitstream::popcount_words_portable(data)
}
