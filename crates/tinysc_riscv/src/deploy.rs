// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — tinySC Deploy Configuration Generator

//! CLI-driven deployment configuration for RISC-V MCU targets.
//!
//! Generates `.cargo/config.toml`, `memory.x` linker scripts, and
//! board-specific `main.rs` stubs for one-command deployment.

/// Supported RISC-V board targets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Board {
    Esp32c3,
    Esp32c6,
    Esp32h2,
    Gd32vf103,
    Ch32v307,
    K210,
    Generic,
}

/// Error returned when a deployment board name is not recognised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseBoardError;

impl Board {
    /// Rust target triple.
    pub const fn target_triple(&self) -> &'static str {
        match self {
            Board::Esp32c3 => "riscv32imc-unknown-none-elf",
            Board::Esp32c6 => "riscv32imac-unknown-none-elf",
            Board::Esp32h2 => "riscv32imc-unknown-none-elf",
            Board::Gd32vf103 => "riscv32imac-unknown-none-elf",
            Board::Ch32v307 => "riscv32imac-unknown-none-elf",
            Board::K210 => "riscv64gc-unknown-none-elf",
            Board::Generic => "riscv32imc-unknown-none-elf",
        }
    }

    /// Human-readable board name.
    pub const fn name(&self) -> &'static str {
        match self {
            Board::Esp32c3 => "ESP32-C3",
            Board::Esp32c6 => "ESP32-C6",
            Board::Esp32h2 => "ESP32-H2",
            Board::Gd32vf103 => "GD32VF103",
            Board::Ch32v307 => "CH32V307",
            Board::K210 => "Kendryte K210",
            Board::Generic => "Generic RISC-V MCU",
        }
    }

    /// RAM in kilobytes.
    pub const fn ram_kb(&self) -> u32 {
        match self {
            Board::Esp32c3 => 400,
            Board::Esp32c6 => 512,
            Board::Esp32h2 => 320,
            Board::Gd32vf103 => 32,
            Board::Ch32v307 => 64,
            Board::K210 => 8192,
            Board::Generic => 64,
        }
    }

    /// Flash in kilobytes.
    pub const fn flash_kb(&self) -> u32 {
        match self {
            Board::Esp32c3 => 4096,
            Board::Esp32c6 => 4096,
            Board::Esp32h2 => 4096,
            Board::Gd32vf103 => 128,
            Board::Ch32v307 => 256,
            Board::K210 => 16384,
            Board::Generic => 256,
        }
    }

    /// Parse from string (e.g., "esp32c6").
    #[allow(
        clippy::should_implement_trait,
        reason = "retain the public Option-returning API alongside core::str::FromStr"
    )]
    pub fn from_str(s: &str) -> Option<Self> {
        // Case-insensitive match without allocating
        let s = s.as_bytes();
        match s.len() {
            7 if eq_ci(s, b"esp32c3") || eq_ci(s, b"esp32-c3") => Some(Board::Esp32c3),
            7 if eq_ci(s, b"esp32c6") || eq_ci(s, b"esp32-c6") => Some(Board::Esp32c6),
            7 if eq_ci(s, b"esp32h2") || eq_ci(s, b"esp32-h2") => Some(Board::Esp32h2),
            8 if eq_ci(s, b"esp32-c3") => Some(Board::Esp32c3),
            8 if eq_ci(s, b"esp32-c6") => Some(Board::Esp32c6),
            8 if eq_ci(s, b"esp32-h2") => Some(Board::Esp32h2),
            9 if eq_ci(s, b"gd32vf103") => Some(Board::Gd32vf103),
            8 if eq_ci(s, b"ch32v307") => Some(Board::Ch32v307),
            4 if eq_ci(s, b"k210") => Some(Board::K210),
            7 if eq_ci(s, b"generic") => Some(Board::Generic),
            _ => None,
        }
    }
}

impl core::str::FromStr for Board {
    type Err = ParseBoardError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Board::from_str(s).ok_or(ParseBoardError)
    }
}

/// Case-insensitive byte comparison.
fn eq_ci(a: &[u8], b: &[u8]) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// Generate `.cargo/config.toml` for the target board.
pub fn cargo_config(board: Board) -> &'static str {
    match board {
        Board::Esp32c3 | Board::Esp32c6 | Board::Esp32h2 => {
            r#"[build]
target = "riscv32imc-unknown-none-elf"

[target.riscv32imc-unknown-none-elf]
rustflags = [
    "-C", "link-arg=-Tmemory.x",
    "-C", "link-arg=-Tlink.x",
]

[unstable]
build-std = ["core"]
"#
        }
        Board::K210 => {
            r#"[build]
target = "riscv64gc-unknown-none-elf"

[target.riscv64gc-unknown-none-elf]
rustflags = [
    "-C", "link-arg=-Tmemory.x",
    "-C", "link-arg=-Tlink.x",
]

[unstable]
build-std = ["core"]
"#
        }
        _ => {
            r#"[build]
target = "riscv32imac-unknown-none-elf"

[target.riscv32imac-unknown-none-elf]
rustflags = [
    "-C", "link-arg=-Tmemory.x",
    "-C", "link-arg=-Tlink.x",
]

[unstable]
build-std = ["core"]
"#
        }
    }
}

/// Generate `memory.x` linker script.
pub fn memory_x(board: Board) -> &'static str {
    match board {
        Board::Esp32c3 | Board::Esp32c6 => {
            r#"/* SC-NeuroCore tinySC — ESP32-C3/C6 memory layout */
MEMORY
{
    IRAM : ORIGIN = 0x40380000, LENGTH = 400K
    DRAM : ORIGIN = 0x3FC80000, LENGTH = 400K
    FLASH : ORIGIN = 0x42000000, LENGTH = 4M
}

REGION_ALIAS("REGION_TEXT", IRAM);
REGION_ALIAS("REGION_RODATA", FLASH);
REGION_ALIAS("REGION_DATA", DRAM);
REGION_ALIAS("REGION_BSS", DRAM);
REGION_ALIAS("REGION_STACK", DRAM);
"#
        }
        Board::Gd32vf103 => {
            r#"/* SC-NeuroCore tinySC — GD32VF103 memory layout */
MEMORY
{
    FLASH : ORIGIN = 0x08000000, LENGTH = 128K
    RAM : ORIGIN = 0x20000000, LENGTH = 32K
}

REGION_ALIAS("REGION_TEXT", FLASH);
REGION_ALIAS("REGION_RODATA", FLASH);
REGION_ALIAS("REGION_DATA", RAM);
REGION_ALIAS("REGION_BSS", RAM);
REGION_ALIAS("REGION_STACK", RAM);
"#
        }
        _ => {
            r#"/* SC-NeuroCore tinySC — Generic RISC-V memory layout */
MEMORY
{
    FLASH : ORIGIN = 0x08000000, LENGTH = 256K
    RAM : ORIGIN = 0x20000000, LENGTH = 64K
}

REGION_ALIAS("REGION_TEXT", FLASH);
REGION_ALIAS("REGION_RODATA", FLASH);
REGION_ALIAS("REGION_DATA", RAM);
REGION_ALIAS("REGION_BSS", RAM);
REGION_ALIAS("REGION_STACK", RAM);
"#
        }
    }
}

/// Generate a `main.rs` stub for bare-metal deployment.
pub fn main_stub(_board: Board, _num_layers: usize, _neurons_per_layer: usize) -> &'static str {
    // Returns a generic template; real codegen would be more dynamic.
    r#"// SC-NeuroCore tinySC — Auto-generated bare-metal main
// Run: sc-neurocore deploy --target riscv-mcu --board <board>

#![no_std]
#![no_main]

use tinysc::network::{NetworkRunner, LayerConfig};

#[no_mangle]
pub extern "C" fn main() -> ! {
    let mut net = NetworkRunner::new();
    net.add_layer(LayerConfig::new(4, 256, 10, 1, 0xACE1));
    net.add_layer(LayerConfig::new(2, 128, 5, 0, 0xBEEF));

    loop {
        let input = read_sensor();
        let spikes = net.tick(input);
        write_output(spikes);
    }
}

#[inline(never)]
fn read_sensor() -> u32 {
    // Board integration point: read from GPIO/ADC.
    42
}

#[inline(never)]
fn write_output(_spikes: u64) {
    // Board integration point: toggle GPIO or send UART.
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_boards_parse() {
        assert_eq!(Board::from_str("esp32c6"), Some(Board::Esp32c6));
        assert_eq!(Board::from_str("esp32-c3"), Some(Board::Esp32c3));
        assert_eq!(Board::from_str("k210"), Some(Board::K210));
        assert_eq!(Board::from_str("unknown"), None);
        assert_eq!("ESP32-C6".parse::<Board>(), Ok(Board::Esp32c6));
        assert_eq!("unknown".parse::<Board>(), Err(ParseBoardError));
    }

    #[test]
    fn test_board_properties() {
        for board in [
            Board::Esp32c3,
            Board::Esp32c6,
            Board::Esp32h2,
            Board::Gd32vf103,
            Board::Ch32v307,
            Board::K210,
            Board::Generic,
        ] {
            assert!(!board.name().is_empty());
            assert!(!board.target_triple().is_empty());
            assert!(board.ram_kb() > 0);
            assert!(board.flash_kb() > 0);
        }
    }

    #[test]
    fn test_k210_is_rv64() {
        assert!(Board::K210.target_triple().contains("riscv64"));
    }

    #[test]
    fn test_esp32_is_rv32() {
        assert!(Board::Esp32c6.target_triple().contains("riscv32"));
    }

    #[test]
    fn test_cargo_config_not_empty() {
        for board in [Board::Esp32c6, Board::K210, Board::Gd32vf103] {
            let cfg = cargo_config(board);
            assert!(cfg.contains("[build]"));
            assert!(cfg.contains("target"));
        }
    }

    #[test]
    fn test_memory_x_not_empty() {
        for board in [Board::Esp32c6, Board::Gd32vf103, Board::Generic] {
            let mem = memory_x(board);
            assert!(mem.contains("MEMORY"));
            assert!(mem.contains("FLASH"));
        }
    }

    #[test]
    fn test_main_stub_valid() {
        let stub = main_stub(Board::Esp32c6, 2, 4);
        assert!(stub.contains("#![no_std]"));
        assert!(stub.contains("NetworkRunner"));
    }
}
