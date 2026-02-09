#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn popcount_neon(data: &[u64]) -> u64 {
    let mut total = 0_u64;
    let mut chunks = data.chunks_exact(2);

    for chunk in &mut chunks {
        let v = vld1q_u8(chunk.as_ptr() as *const u8);
        let masked = vandq_u8(v, vdupq_n_u8(0xff));
        let byte_counts = vcntq_u8(masked);
        let sum16 = vpaddlq_u8(byte_counts);
        let sum32 = vpaddlq_u16(sum16);
        let sum64 = vpaddlq_u32(sum32);
        total += vgetq_lane_u64(sum64, 0) + vgetq_lane_u64(sum64, 1);
    }

    total + crate::bitstream::popcount_words_portable(chunks.remainder())
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn popcount_neon(data: &[u64]) -> u64 {
    crate::bitstream::popcount_words_portable(data)
}
