# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SHD FPGA Demo
"""SC-NeuroCore SHD FPGA Demo — Speech Digit Classification on PYNQ-Z2.

This script demonstrates real-time inference of the Heidelberg Spiking
Digits (SHD) dataset on FPGA using the SC-NeuroCore hardware accelerator.

Prerequisites:
    - PYNQ-Z2 board with sc_shd overlay loaded
    - sc_shd.bit and sc_shd.hwh in /home/xilinx/
    - numpy installed on PYNQ

Model:
    - Architecture: Input(140) -> DCLS -> Dense(140x128) -> Vmin_LIF
                    -> DCLS -> Dense(128x128) -> Vmin_LIF
                    -> Dense(128x20) -> Sum_v -> argmax
    - Accuracy: 75.2% on the SHD test set for the trained floating-point
      model; the hardware path has no measured accuracy, and a bitstream
      from create_block_design.tcl carries no trained weights or delays
      (ARCHITECTURE.md, section "SHD FPGA Pipeline")
    - Training: DCLS max (Hammouamri 2024) with learnable delay sharpening

Usage:
    python demo_shd_fpga.py
"""

import importlib

import numpy as np

# --- PYNQ setup ---
try:
    Overlay = importlib.import_module("pynq").Overlay

    overlay = Overlay("/home/xilinx/sc_shd.bit")
    mmio = overlay.sc_shd_0.mmio  # instance name from create_block_design.tcl
    print("FPGA overlay loaded successfully")
except ImportError:
    print("PYNQ not available — running in simulation mode")
    mmio = None

# --- Import driver ---
from sc_shd_driver import SHDAccelerator

# --- Q16.16 scales from trained dcls_max checkpoint ---
# Source: data/masquelier_shd/fpga_artifacts/dcls_max/scales.json
# Computed as: round(float_scale * 65536)
SCALE_L1_Q16_16 = 0x00000751  # 0.028572 (layer1_input_to_h1)
SCALE_L2_Q16_16 = 0x000007B0  # 0.030024 (layer2_h1_to_h2)
SCALE_L3_Q16_16 = 0x0000011B  # 0.004320 (layer3_h2_output)


def generate_synthetic_spike_raster(
    t_steps: int = 250, n_channels: int = 140, rate: float = 0.05
) -> np.ndarray:
    """Generate a random spike raster for testing.

    Parameters
    ----------
    t_steps : int
        Number of timesteps.
    n_channels : int
        Number of input channels (140 for SHD).
    rate : float
        Spike probability per channel per step.

    Returns
    -------
    np.ndarray
        Binary array of shape (t_steps, n_channels).
    """
    rng = np.random.default_rng(42)
    return (rng.random((t_steps, n_channels)) < rate).astype(np.uint8)


def main() -> None:
    """Run SHD inference demo."""
    if mmio is None:
        print("No FPGA available. Generating synthetic test data...")
        print("To run on hardware, execute on PYNQ-Z2 board.")
        return

    accel = SHDAccelerator(mmio)

    # Set scales
    accel.set_scales(SCALE_L1_Q16_16, SCALE_L2_Q16_16, SCALE_L3_Q16_16)
    print(f"Scales set: L1={SCALE_L1_Q16_16:#x}, L2={SCALE_L2_Q16_16:#x}, L3={SCALE_L3_Q16_16:#x}")

    # Generate test data (replace with real SHD data for accuracy eval)
    spike_raster = generate_synthetic_spike_raster(t_steps=250)
    print(f"Input: {spike_raster.shape[0]} timesteps x {spike_raster.shape[1]} channels")
    print(f"Total spikes: {spike_raster.sum()}")

    # Run inference
    print("Running FPGA inference...")
    predicted = accel.classify(spike_raster)
    print(f"Predicted class: {predicted}")

    # Read all output voltages
    voltages = accel.read_output_voltages()
    print(f"Output voltages: {voltages}")
    print(f"Argmax: {np.argmax(voltages)} (class {np.argmax(voltages)})")

    # Core clock cycles only: each input step also costs six MMIO writes from
    # Python, which this count leaves out. No latency has been measured.
    t_total = spike_raster.shape[0] + 30 + 3  # data + padding + pipeline
    freq_mhz = 100
    core_us = t_total / freq_mhz
    print(
        f"Core clock cycles: {t_total} ({core_us:.1f} us at {freq_mhz} MHz),"
        " excluding the Python MMIO writes; not a measured latency"
    )


if __name__ == "__main__":
    main()
