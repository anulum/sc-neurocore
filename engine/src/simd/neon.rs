#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
/// Count set bits in 64-bit words using ARM NEON instructions.
///
/// # Safety
/// Caller must ensure the current CPU supports `neon`.
pub unsafe fn popcount_neon(data: &[u64]) -> u64 {
    let mut total = 0_u64;
    let mut chunks = data.chunks_exact(2);

    for chunk in &mut chunks {
        let v = vld1q_u8(chunk.as_ptr() as *const u8);
        let byte_counts = vcntq_u8(v);
        let sum16 = vpaddlq_u8(byte_counts);
        let sum32 = vpaddlq_u16(sum16);
        let sum64 = vpaddlq_u32(sum32);
        total += vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
    }

    total + crate::bitstream::popcount_words_portable(chunks.remainder())
}

#[cfg(not(target_arch = "aarch64"))]
/// Fallback popcount when NEON is unavailable on this architecture.
///
/// # Safety
/// This function is marked unsafe for API parity with the NEON variant.
pub unsafe fn popcount_neon(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}
