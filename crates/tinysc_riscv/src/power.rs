// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC Power & Memory Estimation (no_std)

//! Power consumption and memory footprint estimation for RISC-V MCU targets.
//!
//! Enables pre-deployment validation that a network fits in target board
//! RAM/flash and provides µW power estimates at given clock frequencies.

use crate::deploy::Board;

/// Estimated power profile for a target board.
pub struct PowerProfile {
    pub board: Board,
    pub clock_mhz: u32,
    pub active_uw: u32,    // µW during inference
    pub sleep_uw: u32,     // µW in deep sleep
    pub ops_per_tick: u32, // SC bit-ops per network tick
}

impl PowerProfile {
    /// Create a power profile for a board at a given clock frequency.
    pub const fn new(board: Board, clock_mhz: u32) -> Self {
        let (active, sleep) = match board {
            Board::Esp32c3 => (15_000, 5), // ~15 mW active, 5 µW deep sleep
            Board::Esp32c6 => (18_000, 7),
            Board::Esp32h2 => (12_000, 3),
            Board::Gd32vf103 => (8_000, 10),
            Board::Ch32v307 => (10_000, 8),
            Board::K210 => (300_000, 50), // dual-core RV64
            Board::Generic => (10_000, 10),
        };
        // Scale linearly with clock (reference is 160 MHz)
        let scaled_active = (active as u64 * clock_mhz as u64 / 160) as u32;
        Self {
            board,
            clock_mhz,
            active_uw: scaled_active,
            sleep_uw: sleep,
            ops_per_tick: 0,
        }
    }

    /// Estimate µW for a given duty cycle (0.0 = always sleep, 1.0 = always active).
    pub fn duty_cycled_uw(&self, duty: f32) -> u32 {
        let active = self.active_uw as f32 * duty;
        let sleep = self.sleep_uw as f32 * (1.0 - duty);
        (active + sleep) as u32
    }
}

/// Memory footprint estimate for a tinySC network.
pub struct MemoryFootprint {
    pub stack_bytes: usize,
    pub static_bytes: usize,
    pub total_bytes: usize,
    pub fits_in_ram: bool,
    pub fits_in_flash: bool,
}

impl MemoryFootprint {
    /// Estimate memory for a network configuration.
    ///
    /// - `num_layers`: number of layers
    /// - `neurons_per_layer`: max neurons in any layer
    /// - `bs_words`: bitstream words per neuron
    pub fn estimate(
        num_layers: usize,
        neurons_per_layer: usize,
        bs_words: usize,
        board: Board,
    ) -> Self {
        // LifNeuron: 4 + 4 + 1 + 1 + 1 = 11 bytes (padded to 12)
        let neuron_size = 12_usize;
        // LayerState: neurons + Lfsr16 + config + spike_mask + overhead
        let layer_size = neuron_size * neurons_per_layer + 32;
        // NetworkRunner: layers + counters
        let net_size = layer_size * num_layers + 16;
        // Bitstream buffers (temporary, on stack)
        let bs_stack = bs_words * 4; // u32 = 4 bytes

        let stack = net_size + bs_stack + 256; // 256 for call overhead
        let static_code = 8_192; // approximate code size

        let total = stack + static_code;
        let ram_bytes = (board.ram_kb() as usize) * 1024;
        let flash_bytes = (board.flash_kb() as usize) * 1024;

        Self {
            stack_bytes: stack,
            static_bytes: static_code,
            total_bytes: total,
            fits_in_ram: stack <= ram_bytes,
            fits_in_flash: static_code <= flash_bytes,
        }
    }

    /// Maximum neurons that fit in a board's RAM (single layer).
    pub fn max_neurons(board: Board) -> usize {
        let ram = (board.ram_kb() as usize) * 1024;
        let overhead = 512; // stack + runtime overhead
        if ram <= overhead {
            return 0;
        }
        (ram - overhead) / 12
    }
}

/// Watchdog timer configuration.
pub struct WatchdogConfig {
    pub timeout_ms: u32,
    pub enabled: bool,
}

impl WatchdogConfig {
    pub const fn new(timeout_ms: u32) -> Self {
        Self {
            timeout_ms,
            enabled: true,
        }
    }

    pub const fn disabled() -> Self {
        Self {
            timeout_ms: 0,
            enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_profile_creation() {
        let pp = PowerProfile::new(Board::Esp32c6, 160);
        assert_eq!(pp.active_uw, 18_000);
        assert_eq!(pp.sleep_uw, 7);
    }

    #[test]
    fn test_power_scaled_with_clock() {
        let pp80 = PowerProfile::new(Board::Esp32c6, 80);
        let pp160 = PowerProfile::new(Board::Esp32c6, 160);
        assert!(pp80.active_uw < pp160.active_uw);
    }

    #[test]
    fn test_duty_cycled() {
        let pp = PowerProfile::new(Board::Esp32c6, 160);
        let full = pp.duty_cycled_uw(1.0);
        let half = pp.duty_cycled_uw(0.5);
        let sleep = pp.duty_cycled_uw(0.0);
        assert!(full > half);
        assert!(half > sleep);
    }

    #[test]
    fn test_memory_footprint_fits() {
        let fp = MemoryFootprint::estimate(2, 16, 8, Board::Esp32c6);
        assert!(fp.fits_in_ram, "should fit in 512KB RAM");
        assert!(fp.fits_in_flash);
        assert!(fp.total_bytes > 0);
    }

    #[test]
    fn test_memory_footprint_tiny_board() {
        let fp = MemoryFootprint::estimate(8, 64, 16, Board::Gd32vf103);
        // GD32VF103 has only 32KB RAM — might be tight
        assert!(fp.total_bytes > 0);
    }

    #[test]
    fn test_max_neurons() {
        let n_esp = MemoryFootprint::max_neurons(Board::Esp32c6);
        let n_gd = MemoryFootprint::max_neurons(Board::Gd32vf103);
        assert!(n_esp > n_gd); // more RAM = more neurons
        assert!(n_esp > 0);
    }

    #[test]
    fn test_all_boards_power() {
        for board in [
            Board::Esp32c3,
            Board::Esp32c6,
            Board::Esp32h2,
            Board::Gd32vf103,
            Board::Ch32v307,
            Board::K210,
            Board::Generic,
        ] {
            let pp = PowerProfile::new(board, 160);
            assert!(pp.active_uw > 0);
            assert!(pp.sleep_uw > 0);
        }
    }

    #[test]
    fn test_watchdog_config() {
        let wdt = WatchdogConfig::new(5000);
        assert!(wdt.enabled);
        assert_eq!(wdt.timeout_ms, 5000);

        let dis = WatchdogConfig::disabled();
        assert!(!dis.enabled);
    }
}
