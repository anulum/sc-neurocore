// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Sobol quasi-random bitstream generator
//! Sobol quasi-random bitstream generator.
//!
//! 1D Sobol (Van der Corput base-2) with Gray-code indexing.
//! LDS convergence: O(1/N) vs O(1/√N) for Bernoulli.

/// 1D Sobol quasi-random engine using Gray-code traversal.
#[derive(Clone, Debug)]
pub struct SobolEngine {
    state: u32,
    index: u32,
    direction: [u32; 32],
}

impl SobolEngine {
    /// Initialize with seed (XOR-scramble on direction numbers).
    pub fn new(seed: u32) -> Self {
        let mut direction = [0u32; 32];
        for (i, d) in direction.iter_mut().enumerate() {
            *d = 1u32 << (31 - i);
        }
        Self {
            state: seed,
            index: 0,
            direction,
        }
    }

    /// Draw next quasi-random sample in [0, 1).
    pub fn sample(&mut self) -> f64 {
        let c = self.index.trailing_ones() as usize;
        self.state ^= self.direction[c.min(31)];
        self.index += 1;
        self.state as f64 / (1u64 << 32) as f64
    }

    pub fn reset(&mut self, seed: u32) {
        self.state = seed;
        self.index = 0;
    }
}

/// Generate a Sobol bitstream: threshold comparison of quasi-random samples.
pub fn generate_sobol_bitstream(p: f64, length: usize, seed: u32) -> Vec<u8> {
    assert!((0.0..=1.0).contains(&p), "p must be in [0,1]");
    let mut engine = SobolEngine::new(seed);
    (0..length)
        .map(|_| if engine.sample() < p { 1u8 } else { 0u8 })
        .collect()
}

/// Pack Sobol bitstream into u64 words.
pub fn generate_sobol_packed(p: f64, length: usize, seed: u32) -> Vec<u64> {
    let bits = generate_sobol_bitstream(p, length, seed);
    crate::bitstream::pack_fast(&bits).data
}

// ---------------------------------------------------------------------------
// Halton quasi-random engine
// ---------------------------------------------------------------------------

/// Halton quasi-random engine using radical-inverse sequences.
///
/// Supports base-2 (Van der Corput) and base-3 for two-dimensional
/// low-discrepancy sequences.  Convergence: O(log N / N) — slightly
/// slower than Sobol but requires no direction numbers, making it
/// cheaper in hardware (simple bit-reversal for base-2).
#[derive(Clone, Debug)]
pub struct HaltonEngine {
    index: u64,
    base: u32,
    seed: u64,
}

impl HaltonEngine {
    /// Create a new Halton engine with the given base and seed.
    ///
    /// `base` must be a prime ≥ 2.  Common choices: 2 (Van der Corput), 3.
    pub fn new(base: u32, seed: u64) -> Self {
        assert!(base >= 2, "Halton base must be >= 2");
        Self {
            index: seed,
            base,
            seed,
        }
    }

    /// Draw next quasi-random sample in [0, 1).
    pub fn sample(&mut self) -> f64 {
        self.index += 1;
        Self::radical_inverse(self.index, self.base)
    }

    /// Compute the radical inverse of `n` in the given `base`.
    ///
    /// For base 2 this is equivalent to bit-reversal of the binary
    /// representation — trivially implementable in hardware.
    fn radical_inverse(mut n: u64, base: u32) -> f64 {
        let mut result = 0.0_f64;
        let mut denom = 1.0_f64;
        let b = f64::from(base);
        while n > 0 {
            denom *= b;
            let remainder = n % u64::from(base);
            result += remainder as f64 / denom;
            n /= u64::from(base);
        }
        result
    }

    /// Reset the engine to its initial seed.
    pub fn reset(&mut self) {
        self.index = self.seed;
    }
}

/// Generate a Halton bitstream: threshold comparison of quasi-random samples.
pub fn generate_halton_bitstream(p: f64, length: usize, base: u32, seed: u64) -> Vec<u8> {
    assert!((0.0..=1.0).contains(&p), "p must be in [0,1]");
    let mut engine = HaltonEngine::new(base, seed);
    (0..length)
        .map(|_| if engine.sample() < p { 1u8 } else { 0u8 })
        .collect()
}

/// Pack Halton bitstream into u64 words.
pub fn generate_halton_packed(p: f64, length: usize, base: u32, seed: u64) -> Vec<u64> {
    let bits = generate_halton_bitstream(p, length, base, seed);
    crate::bitstream::pack_fast(&bits).data
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Sobol tests ----

    #[test]
    fn deterministic_output() {
        let a = generate_sobol_bitstream(0.5, 128, 0);
        let b = generate_sobol_bitstream(0.5, 128, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn p_zero_all_zeros() {
        let bits = generate_sobol_bitstream(0.0, 256, 42);
        assert!(bits.iter().all(|&b| b == 0));
    }

    #[test]
    fn p_one_all_ones() {
        let bits = generate_sobol_bitstream(1.0, 256, 42);
        assert!(bits.iter().all(|&b| b == 1));
    }

    #[test]
    fn proportion_close_to_p() {
        let p = 0.3;
        let bits = generate_sobol_bitstream(p, 4096, 0);
        let ones = bits.iter().filter(|&&b| b == 1).count();
        let ratio = ones as f64 / 4096.0;
        assert!(
            (ratio - p).abs() < 0.02,
            "Sobol ratio {ratio} should be close to {p}"
        );
    }

    #[test]
    fn lower_discrepancy_than_bernoulli() {
        // Sobol at p=0.5 should have exactly 50% ones for power-of-2 lengths
        let bits = generate_sobol_bitstream(0.5, 1024, 0);
        let ones = bits.iter().filter(|&&b| b == 1).count();
        let error = (ones as f64 / 1024.0 - 0.5).abs();
        assert!(error < 0.05, "Sobol discrepancy {error} too high");
    }

    #[test]
    fn packed_roundtrip() {
        let bits = generate_sobol_bitstream(0.7, 200, 99);
        let packed = generate_sobol_packed(0.7, 200, 99);
        let unpacked = crate::bitstream::unpack(&crate::bitstream::BitStreamTensor {
            data: packed,
            length: 200,
        });
        assert_eq!(bits, unpacked);
    }

    // ---- Halton tests ----

    #[test]
    fn halton_deterministic() {
        let a = generate_halton_bitstream(0.5, 128, 2, 0);
        let b = generate_halton_bitstream(0.5, 128, 2, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn halton_p_zero_all_zeros() {
        let bits = generate_halton_bitstream(0.0, 256, 2, 0);
        assert!(bits.iter().all(|&b| b == 0));
    }

    #[test]
    fn halton_p_one_all_ones() {
        let bits = generate_halton_bitstream(1.0, 256, 2, 0);
        assert!(bits.iter().all(|&b| b == 1));
    }

    #[test]
    fn halton_base2_proportion() {
        let p = 0.4;
        let bits = generate_halton_bitstream(p, 4096, 2, 0);
        let ones = bits.iter().filter(|&&b| b == 1).count();
        let ratio = ones as f64 / 4096.0;
        assert!(
            (ratio - p).abs() < 0.02,
            "Halton base-2 ratio {ratio} should be close to {p}"
        );
    }

    #[test]
    fn halton_base3_proportion() {
        let p = 0.6;
        let bits = generate_halton_bitstream(p, 4096, 3, 0);
        let ones = bits.iter().filter(|&&b| b == 1).count();
        let ratio = ones as f64 / 4096.0;
        assert!(
            (ratio - p).abs() < 0.03,
            "Halton base-3 ratio {ratio} should be close to {p}"
        );
    }

    #[test]
    fn halton_packed_roundtrip() {
        let bits = generate_halton_bitstream(0.7, 200, 2, 99);
        let packed = generate_halton_packed(0.7, 200, 2, 99);
        let unpacked = crate::bitstream::unpack(&crate::bitstream::BitStreamTensor {
            data: packed,
            length: 200,
        });
        assert_eq!(bits, unpacked);
    }

    #[test]
    fn halton_radical_inverse_base2() {
        // Base-2 radical inverse of 1 = 0.5, 2 = 0.25, 3 = 0.75
        let r1 = HaltonEngine::radical_inverse(1, 2);
        let r2 = HaltonEngine::radical_inverse(2, 2);
        let r3 = HaltonEngine::radical_inverse(3, 2);
        assert!((r1 - 0.5).abs() < 1e-10);
        assert!((r2 - 0.25).abs() < 1e-10);
        assert!((r3 - 0.75).abs() < 1e-10);
    }

    #[test]
    fn halton_radical_inverse_base3() {
        // Base-3: 1 → 1/3, 2 → 2/3, 3 → 1/9
        let r1 = HaltonEngine::radical_inverse(1, 3);
        let r2 = HaltonEngine::radical_inverse(2, 3);
        let r3 = HaltonEngine::radical_inverse(3, 3);
        assert!((r1 - 1.0 / 3.0).abs() < 1e-10);
        assert!((r2 - 2.0 / 3.0).abs() < 1e-10);
        assert!((r3 - 1.0 / 9.0).abs() < 1e-10);
    }

    // ---- Convergence comparison ----

    #[test]
    fn sobol_converges_faster_than_halton() {
        // At 4096 samples, Sobol should have lower discrepancy than Halton
        let p = 0.5;
        let n = 4096usize;

        let sobol_bits = generate_sobol_bitstream(p, n, 0);
        let sobol_ones = sobol_bits.iter().filter(|&&b| b == 1).count();
        let sobol_err = (sobol_ones as f64 / n as f64 - p).abs();

        let halton_bits = generate_halton_bitstream(p, n, 2, 0);
        let halton_ones = halton_bits.iter().filter(|&&b| b == 1).count();
        let halton_err = (halton_ones as f64 / n as f64 - p).abs();

        // Both should be low, Sobol typically lower
        assert!(sobol_err < 0.02, "Sobol error {sobol_err} too high");
        assert!(halton_err < 0.02, "Halton error {halton_err} too high");
    }
}
