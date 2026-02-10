//! # Fixed-Point LIF Neuron
//!
//! Integer LIF neuron model used by the v3 engine for deterministic,
//! hardware-friendly stochastic-computing experiments.

/// Mask and sign-interpret an integer to `width` bits.
#[inline]
pub fn mask(value: i32, width: u32) -> i16 {
    let m = (1_i64 << width) - 1;
    let mut v = (value as i64) & m;
    if v >= (1_i64 << (width - 1)) {
        v -= 1_i64 << width;
    }
    if width >= 32 {
        // For extended intermediate masks (e.g. 2 * data_width), emulate
        // Python's signed interpretation before truncating to i16.
        v as i32 as i16
    } else {
        v as i16
    }
}

/// Fixed-point leaky-integrate-and-fire neuron state and parameters.
#[derive(Clone, Debug)]
pub struct FixedPointLif {
    /// Membrane potential.
    pub v: i16,
    /// Refractory counter in simulation steps.
    pub refractory_counter: i32,
    /// Arithmetic data width.
    pub data_width: u32,
    /// Fraction bits for fixed-point scaling.
    pub fraction: u32,
    /// Resting potential.
    pub v_rest: i16,
    /// Reset potential after spike.
    pub v_reset: i16,
    /// Spike threshold.
    pub v_threshold: i16,
    /// Refractory period length in steps.
    pub refractory_period: i32,
}

impl FixedPointLif {
    /// Construct a fixed-point LIF neuron.
    pub fn new(
        data_width: u32,
        fraction: u32,
        v_rest: i16,
        v_reset: i16,
        v_threshold: i16,
        refractory_period: i32,
    ) -> Self {
        Self {
            v: v_rest,
            refractory_counter: 0,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        }
    }

    /// Advance one simulation step.
    ///
    /// Returns `(spike, membrane_voltage)`.
    pub fn step(&mut self, leak_k: i16, gain_k: i16, i_t: i16, noise_in: i16) -> (i32, i16) {
        let w = self.data_width;
        let diff = mask((self.v_rest as i32) - (self.v as i32), 2 * w) as i32;
        let leak_mul = diff * (leak_k as i32);
        let dv_leak = mask(leak_mul >> self.fraction, self.data_width);

        let in_mul = (i_t as i32) * (gain_k as i32);
        let dv_in = mask(in_mul >> self.fraction, self.data_width);

        let v_next = mask(
            (self.v as i32) + (dv_leak as i32) + (dv_in as i32) + (noise_in as i32),
            self.data_width,
        );

        let mut spike = if v_next >= self.v_threshold {
            self.v = self.v_reset;
            self.refractory_counter = self.refractory_period;
            1
        } else {
            self.v = v_next;
            0
        };

        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            self.v = self.v_rest;
            spike = 0;
        }

        (spike, mask(self.v as i32, w))
    }

    /// Reset internal state to resting potential.
    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory_counter = 0;
    }
}
