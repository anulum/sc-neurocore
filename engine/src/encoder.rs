//! # Stochastic Encoders
//!
//! LFSR-driven Bernoulli encoding utilities used by SC-NeuroCore kernels.

/// 16-bit linear-feedback shift register used as pseudo-random source.
#[derive(Clone, Debug)]
pub struct Lfsr16 {
    /// Current register value.
    pub reg: u16,
    /// Register width in bits.
    pub width: u32,
}

impl Lfsr16 {
    /// Create an LFSR with a non-zero seed.
    pub fn new(seed: u16) -> Self {
        assert_ne!(seed, 0, "LFSR seed must be non-zero.");
        Self {
            reg: seed,
            width: 16,
        }
    }

    /// Advance one LFSR step and return the new register value.
    pub fn step(&mut self) -> u16 {
        let feedback =
            ((self.reg >> 15) ^ (self.reg >> 13) ^ (self.reg >> 12) ^ (self.reg >> 10)) & 1;
        self.reg = (self.reg << 1) | feedback;
        self.reg
    }

    /// Reset register state. If `seed` is `None`, keeps current seed value.
    pub fn reset(&mut self, seed: Option<u16>) {
        let next = seed.unwrap_or(self.reg);
        assert_ne!(next, 0, "LFSR seed must be non-zero.");
        self.reg = next;
    }
}

/// Comparator-based stochastic bitstream encoder.
#[derive(Clone, Debug)]
pub struct BitstreamEncoder {
    /// Underlying LFSR source.
    pub lfsr: Lfsr16,
    /// Data path width for compatibility with fixed-point callers.
    pub data_width: u32,
    seed_init: u16,
}

impl BitstreamEncoder {
    /// Create a new encoder with deterministic seed.
    pub fn new(data_width: u32, seed: u16) -> Self {
        Self {
            lfsr: Lfsr16::new(seed),
            data_width,
            seed_init: seed,
        }
    }

    /// Emit one stochastic bit by comparing RNG value against `x_value`.
    pub fn step(&mut self, x_value: u16) -> u8 {
        self.lfsr.step();
        if self.lfsr.reg < x_value {
            1
        } else {
            0
        }
    }

    /// Reset the encoder LFSR state.
    pub fn reset(&mut self, seed: Option<u16>) {
        let next = seed.unwrap_or(self.seed_init);
        self.lfsr = Lfsr16::new(next);
    }
}
