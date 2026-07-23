# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCompare from former test_profiler_platform.py

"""Focused suite: TestCompare from former test_profiler_platform.py."""

from __future__ import annotations

from tests.profiler_platform_support import *  # noqa: F403

class TestCompare:
    def test_default_platforms(self):
        results = compare(
            layer_sizes=[(16, 8)],
            duration=0.1,
            dt=0.001,
        )
        assert len(results) == 4
        platforms = {r.platform for r in results}
        assert "python" in platforms

    def test_specific_platforms(self):
        results = compare(
            layer_sizes=[(16, 8)],
            platforms=["python"],
        )
        assert len(results) == 1
        assert results[0].platform == "python"

    def test_python_estimates(self):
        results = compare(
            layer_sizes=[(100, 50)],
            duration=0.1,
            dt=0.001,
            platforms=["python"],
        )
        r = results[0]
        assert r.latency_ms > 0
        assert r.throughput_inf_per_s > 0
        assert r.power_mw > 0
        assert r.energy_per_inf_nj > 0

    def test_rust_platform(self):
        results = compare(
            layer_sizes=[(16, 8)],
            platforms=["rust"],
        )
        r = results[0]
        assert r.platform == "rust"
        assert r.latency_ms > 0

    def test_fpga_platform(self):
        results = compare(
            layer_sizes=[(16, 8)],
            platforms=["fpga_ice40"],
        )
        r = results[0]
        assert r.platform == "fpga_ice40"
        assert r.latency_ms > 0

    def test_unknown_platform(self):
        results = compare(
            layer_sizes=[(16, 8)],
            platforms=["quantum_computer"],
        )
        r = results[0]
        assert r.available is False
        assert "Unknown" in r.notes

    def test_sorted_by_energy(self):
        results = compare(
            layer_sizes=[(32, 16)],
            platforms=["python", "fpga_artix7"],
        )
        available = [r for r in results if r.available]
        for i in range(len(available) - 1):
            assert available[i].energy_per_inf_nj <= available[i + 1].energy_per_inf_nj
