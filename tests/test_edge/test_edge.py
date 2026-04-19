# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for edge deployment module

"""Tests for edge power estimation, Sobol generator, and deploy config."""

from sc_neurocore.edge import Board, SobolGenerator
from sc_neurocore.edge.power_estimator import PowerProfile, MemoryFootprint
from sc_neurocore.edge.deploy import generate_cargo_config, generate_memory_x


class TestPowerProfile:
    def test_creation(self):
        pp = PowerProfile.for_board(Board.ESP32_C6, 160)
        assert pp.active_uw == 18_000
        assert pp.sleep_uw == 7

    def test_scaled_with_clock(self):
        pp80 = PowerProfile.for_board(Board.ESP32_C6, 80)
        pp160 = PowerProfile.for_board(Board.ESP32_C6, 160)
        assert pp80.active_uw < pp160.active_uw

    def test_duty_cycled(self):
        pp = PowerProfile.for_board(Board.ESP32_C6, 160)
        full = pp.duty_cycled_uw(1.0)
        half = pp.duty_cycled_uw(0.5)
        sleep = pp.duty_cycled_uw(0.0)
        assert full > half > sleep

    def test_all_boards(self):
        for board in Board:
            pp = PowerProfile.for_board(board, 160)
            assert pp.active_uw > 0
            assert pp.sleep_uw > 0


class TestMemoryFootprint:
    def test_fits_in_ram(self):
        fp = MemoryFootprint.estimate(2, 16, 8, Board.ESP32_C6)
        assert fp.fits_in_ram
        assert fp.fits_in_flash
        assert fp.total_bytes > 0

    def test_max_neurons(self):
        n_esp = MemoryFootprint.max_neurons(Board.ESP32_C6)
        n_gd = MemoryFootprint.max_neurons(Board.GD32VF103)
        assert n_esp > n_gd > 0


class TestSobolGenerator:
    def test_deterministic(self):
        a = SobolGenerator(seed=0)
        b = SobolGenerator(seed=0)
        for _ in range(100):
            assert a.step() == b.step()

    def test_unique_values(self):
        s = SobolGenerator()
        values = {s.step() for _ in range(1000)}
        assert len(values) > 500

    def test_encode_probability(self):
        s = SobolGenerator()
        bs = s.encode(32768, 10000)  # ~50% threshold
        popcount = sum(bin(int(w)).count("1") for w in bs)
        p = popcount / 10000
        assert abs(p - 0.5) < 0.05

    def test_different_seeds(self):
        a = SobolGenerator(seed=0x1234)
        b = SobolGenerator(seed=0x5678)
        assert a.step() != b.step()

    def test_reset(self):
        s = SobolGenerator()
        first = [s.step() for _ in range(10)]
        s.reset()
        second = [s.step() for _ in range(10)]
        assert first == second


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
