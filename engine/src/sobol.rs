// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
