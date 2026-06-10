# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for RISC-V SoC drivers

"""Tests for RISC-V C driver generation."""

from __future__ import annotations

from sc_neurocore.compiler.riscv_driver import generate_riscv_driver


class TestRiscvDriver:
    """Test RISC-V driver generation."""

    def test_baremetal_driver(self) -> None:
        """Should produce a valid bare-metal C driver."""
        c = generate_riscv_driver("sc_lif", {"tau": 16, "vth": 16})
        assert "SC_LIF_BASE" in c
        assert "MMIO_WR" in c
        assert "MMIO_RD" in c
        assert "sc_lif_enable" in c
        assert "sc_lif_reset" in c
        assert "sc_lif_set_tau" in c
        assert "sc_lif_set_vth" in c

    def test_freertos_integration(self) -> None:
        """Should include FreeRTOS task templates."""
        c = generate_riscv_driver("sc_lif", {"tau": 16}, rtos="freertos")
        assert "FreeRTOS.h" in c
        assert "xTaskCreate" in c
        assert "sc_lif_tick" in c

    def test_zephyr_integration(self) -> None:
        """Should include Zephyr thread templates."""
        c = generate_riscv_driver("sc_lif", {"tau": 16}, rtos="zephyr")
        assert "zephyr/kernel.h" in c
        assert "K_THREAD_DEFINE" in c

    def test_custom_base_address(self) -> None:
        """Base address should propagate."""
        c = generate_riscv_driver("sc_lif", {}, base_address=0x8000_0000)
        assert "0x80000000U" in c

    def test_multi_param(self) -> None:
        """Multiple parameters should result in multiple setters."""
        c = generate_riscv_driver("sc_lif", {"tau": 16, "vth": 16, "leak": 16})
        assert "sc_lif_set_tau" in c
        assert "sc_lif_set_vth" in c
        assert "sc_lif_set_leak" in c
