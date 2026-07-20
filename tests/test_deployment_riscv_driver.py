# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — RISC-V deployment driver contracts

"""Contracts for generated RISC-V deployment drivers."""

from __future__ import annotations


class TestRISCVDriver:
    """Tests for RISC-V C driver generation."""

    def test_baremetal_driver(self) -> None:
        """Baremetal driver has MMIO macros and functions."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {"tau": 16, "vth": 16})
        assert "#ifndef SC_LIF_RISCV_H" in c
        assert "MMIO_WR" in c
        assert "MMIO_RD" in c
        assert "sc_lif_enable" in c
        assert "sc_lif_reset" in c
        assert "sc_lif_set_current" in c
        assert "sc_lif_read_current" in c
        assert "sc_lif_get_spikes" in c
        assert "sc_lif_encode" in c
        assert "sc_lif_set_tau" in c
        assert "sc_lif_set_vth" in c
        assert "volatile" in c

    def test_freertos_template(self) -> None:
        """FreeRTOS template includes task and timer."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {"tau": 16}, rtos="freertos")
        assert "FreeRTOS.h" in c
        assert "xTaskCreate" in c
        assert "sc_lif_tick" in c
        assert "sc_lif_start_rtos" in c
        assert "vTaskDelay" in c
        assert "float I = sc_lif_read_current();" in c
        assert "TODO" not in c

    def test_zephyr_template(self) -> None:
        """Zephyr template includes thread and K_THREAD_DEFINE."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {"tau": 16}, rtos="zephyr")
        assert "zephyr/kernel.h" in c
        assert "K_THREAD_DEFINE" in c
        assert "k_msleep" in c
        assert "sc_lif_set_current(sc_lif_read_current())" in c

    def test_custom_base_address(self) -> None:
        """Custom base address propagates."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {}, base_address=0x8000_0000)
        assert "0x80000000" in c

    def test_param_registers(self) -> None:
        """Per-parameter register definitions."""
        from sc_neurocore.compiler.deployment import generate_riscv_driver

        c = generate_riscv_driver("sc_lif", {"tau": 16, "vth": 16, "leak": 16})
        assert "SC_LIF_TAU" in c
        assert "SC_LIF_VTH" in c
        assert "SC_LIF_LEAK" in c
