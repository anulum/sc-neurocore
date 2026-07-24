# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeploy from former test_edge.py

"""Focused suite: TestDeploy from former test_edge.py."""

from __future__ import annotations

from edge_support import *  # noqa: F403


class TestDeploy:
    def test_cargo_config_contains_target(self):
        for board in [Board.ESP32_C6, Board.K210, Board.GD32VF103]:
            cfg = generate_cargo_config(board)
            assert "[build]" in cfg
            assert "target" in cfg

    def test_memory_x_contains_memory(self):
        for board in [Board.ESP32_C3, Board.GD32VF103, Board.GENERIC]:
            mem = generate_memory_x(board)
            assert "MEMORY" in mem
            assert "FLASH" in mem

    def test_k210_is_rv64(self):
        cfg = generate_cargo_config(Board.K210)
        assert "riscv64" in cfg

    def test_esp32_h2_reuses_esp32_c3_cargo_profile(self):
        h2_cfg = generate_cargo_config(Board.ESP32_H2)
        c3_cfg = generate_cargo_config(Board.ESP32_C3)
        assert h2_cfg == c3_cfg

    def test_esp32_c6_and_h2_use_esp32_c3_memory_layout(self):
        c3_mem = generate_memory_x(Board.ESP32_C3)
        c6_mem = generate_memory_x(Board.ESP32_C6)
        h2_mem = generate_memory_x(Board.ESP32_H2)
        assert c6_mem == c3_mem
        assert h2_mem == c3_mem

    def test_generic_board_uses_default_cargo_fallback(self):
        cfg = generate_cargo_config(Board.GENERIC)
        assert 'target = "riscv32imac-unknown-none-elf"' in cfg

    def test_unknown_memory_board_uses_generic_memory_layout(self):
        mem = generate_memory_x(Board.CH32V307)
        assert "LENGTH = 256K" in mem
        assert "LENGTH = 64K" in mem
