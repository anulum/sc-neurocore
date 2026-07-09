# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.profiler (platform comparison)

from __future__ import annotations


from sc_neurocore.profiler import compare, PlatformResult
from sc_neurocore.profiler.platform_profiler import format_table


class TestPlatformResult:
    def test_fields(self):
        r = PlatformResult(
            platform="python",
            latency_ms=10.0,
            throughput_inf_per_s=100.0,
            power_mw=10000.0,
            energy_per_inf_nj=100000.0,
        )
        assert r.available is True
        assert r.notes == ""

    def test_unavailable(self):
        r = PlatformResult(
            platform="custom",
            latency_ms=0,
            throughput_inf_per_s=0,
            power_mw=0,
            energy_per_inf_nj=0,
            available=False,
            notes="Not installed",
        )
        assert r.available is False


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


class TestFormatTable:
    def test_table_format(self):
        results = compare(layer_sizes=[(16, 8)], platforms=["python"])
        table = format_table(results)
        assert "Platform" in table
        assert "python" in table
        assert "(ms)" in table

    def test_unavailable_in_table(self):
        r = PlatformResult(
            platform="missing",
            latency_ms=0,
            throughput_inf_per_s=0,
            power_mw=0,
            energy_per_inf_nj=0,
            available=False,
            notes="N/A",
        )
        table = format_table([r])
        assert "N/A" in table
