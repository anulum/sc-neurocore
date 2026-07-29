// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — TinySC RISC-V Runtime
// Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)

//! # tinySC — Bare-Metal RISC-V Stochastic Computing Runtime
//!
//! A `no_std` runtime that brings the full SC-NeuroCore engine to
//! RV32/RV64 MCUs with optional RVV SIMD and custom popcount instructions.
//!
//! ## Modules
//!
//! - [`bitstream`]: Packed u32-word SC arithmetic (AND, MUX, popcount, SCC)
//! - [`lfsr`]: Deterministic LFSR-16 encoder (bit-compatible with core_engine)
//! - [`neuron`]: LIF and Izhikevich neurons (SC-domain, no FPU)
//! - [`network`]: Fixed-capacity NetworkRunner (stack-only, no heap)
//! - [`ecc`]: Hamming(7,4) ECC (bit-compatible with ScDoctor)
//! - [`deploy`]: Board configuration + linker script generator
//!
//! ## Feature flags
//!
//! - **`std`** (default in workspace): Enables Vec/Box-based dynamic allocation.
//! - **`rvv`**: Enables inline RVV vector intrinsics.
//! - **`custom-popcount`**: Uses CSR-mapped hardware popcount unit.

#![cfg_attr(not(test), no_std)]

pub mod bitstream;
pub mod deploy;
pub mod ecc;
pub mod lfsr;
pub mod network;
pub mod neuron;
pub mod power;
pub mod sobol;
pub mod telemetry;
pub mod weights;

// ── Legacy API (preserved for backward compatibility) ───────────────

/// Fixed-point Q16.16 LIF neuron (original tinySC API).
pub struct TinyLIF {
    pub v: i32,         // Fixed-point Q16.16
    pub threshold: i32, // Fixed-point Q16.16
    pub leak: i32,      // Fixed-point Q0.16 (unsigned)
}

impl TinyLIF {
    #[inline(always)]
    pub fn step(&mut self, input: i32) -> bool {
        let decayed_v = ((self.v as i64 * self.leak as i64) >> 16) as i32;
        self.v = decayed_v.saturating_add(input);

        if self.v >= self.threshold {
            self.v = 0;
            return true;
        }
        false
    }
}

/// Const-generic network runner (original tinySC API).
pub struct LegacyNetworkRunner<const N: usize> {
    pub neurons: [TinyLIF; N],
}

impl<const N: usize> LegacyNetworkRunner<N> {
    pub fn new(threshold: i32, leak: i32) -> Self {
        let neurons = core::array::from_fn(|_| TinyLIF {
            v: 0,
            threshold,
            leak,
        });
        Self { neurons }
    }

    pub fn process_layer(&mut self, inputs: &[i32]) -> [bool; N] {
        let mut outputs = [false; N];
        for (i, output) in outputs.iter_mut().enumerate() {
            let mut acc = 0;
            for &val in inputs {
                acc += val;
            }
            *output = self.neurons[i].step(acc);
        }
        outputs
    }
}

/// C-FFI entry point for firmware.
#[no_mangle]
pub extern "C" fn tinysc_process_step(_input_ptr: *const i32, _n_inputs: usize) -> u32 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lif(threshold: i32, leak: i32) -> TinyLIF {
        TinyLIF {
            v: 0,
            threshold,
            leak,
        }
    }

    #[test]
    fn test_subthreshold_no_spike() {
        let mut n = lif(65536, 32768);
        assert!(!n.step(32768));
        assert_eq!(n.v, 32768);
    }

    #[test]
    fn test_threshold_fires() {
        let mut n = lif(65536, 65536);
        assert!(!n.step(32768));
        assert!(n.step(32768));
    }

    #[test]
    fn test_reset_to_zero() {
        let mut n = lif(65536, 65536);
        n.v = 65535;
        assert!(n.step(10));
        assert_eq!(n.v, 0);
    }

    #[test]
    fn test_decay_halves_potential() {
        let mut n = lif(i32::MAX, 32768);
        n.step(65536);
        assert_eq!(n.v, 65536);
        n.step(0);
        assert_eq!(n.v, 32768);
        n.step(0);
        assert_eq!(n.v, 16384);
    }

    #[test]
    fn test_zero_leak_kills_memory() {
        let mut n = lif(i32::MAX, 0);
        n.step(65536);
        assert_eq!(n.v, 65536);
        n.step(0);
        assert_eq!(n.v, 0);
    }

    #[test]
    fn test_negative_input() {
        let mut n = lif(65536, 65536);
        n.step(32768);
        n.step(-32768);
        assert_eq!(n.v, 0);
    }

    #[test]
    fn test_overflow_safety() {
        let mut n = lif(i32::MAX, 65536);
        n.v = i32::MAX - 1;
        let fired = n.step(100);
        assert!(fired);
        assert_eq!(n.v, 0);
    }

    #[test]
    fn test_legacy_network_runner_new() {
        let runner: LegacyNetworkRunner<4> = LegacyNetworkRunner::new(65536, 32768);
        for neuron in &runner.neurons {
            assert_eq!(neuron.v, 0);
            assert_eq!(neuron.threshold, 65536);
            assert_eq!(neuron.leak, 32768);
        }
    }

    #[test]
    fn test_legacy_network_runner_process_layer() {
        let mut runner: LegacyNetworkRunner<3> = LegacyNetworkRunner::new(65536, 65536);
        let inputs = [65536i32];
        let out = runner.process_layer(&inputs);
        assert_eq!(out, [true, true, true]);
    }

    #[test]
    fn test_legacy_network_runner_subthreshold_layer() {
        let mut runner: LegacyNetworkRunner<2> = LegacyNetworkRunner::new(65536, 32768);
        let inputs = [16384i32];
        let out = runner.process_layer(&inputs);
        assert_eq!(out, [false, false]);
    }
}
