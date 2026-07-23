# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetworkRunner from former test_rust_integration.py

"""Focused suite: TestNetworkRunner from former test_rust_integration.py."""

from __future__ import annotations

from tests.rust_integration_support import *  # noqa: F403

class TestNetworkRunner:
    def test_create_and_run(self) -> None:
        r = engine.NetworkRunner()
        idx = r.add_population("Izhikevich", 10)
        assert idx == 0
        results = r.run(100)
        assert "spike_data" in results
        assert "voltages" in results
        assert "spike_counts" in results

    def test_multiple_populations(self) -> None:
        r = engine.NetworkRunner()
        i0 = r.add_population("Izhikevich", 5)
        i1 = r.add_population("AdEx", 5)
        assert i0 == 0
        assert i1 == 1
        results = r.run(50)
        assert len(results["spike_data"]) == 2
        assert len(results["voltages"]) == 2

    def test_spike_data_u64_format(self) -> None:
        r = engine.NetworkRunner()
        r.add_population("Lapicque", 20)
        results = r.run(200)
        for packed in results["spike_data"][0]:
            nid = int(packed >> 32)
            t = int(packed & 0xFFFFFFFF)
            assert 0 <= nid < 20
            assert 0 <= t < 200

    def test_voltages_returned(self) -> None:
        r = engine.NetworkRunner()
        r.add_population("HodgkinHuxley", 3)
        results = r.run(50)
        v = results["voltages"][0]
        assert len(v) == 3

    def test_projection_csr(self) -> None:
        r = engine.NetworkRunner()
        r.add_population("Izhikevich", 3)
        r.add_population("Izhikevich", 3)
        # All-to-all CSR: 3 source → 3 target, weight 0.5
        row_offsets = [0, 3, 6, 9]
        col_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        values = [0.5] * 9
        r.add_projection(0, 1, row_offsets, col_indices, values)
        results = r.run(100)
        assert len(results["spike_data"]) == 2

    def test_spike_counts(self) -> None:
        r = engine.NetworkRunner()
        r.add_population("Izhikevich", 10)
        results = r.run(100)
        counts = results["spike_counts"]
        assert len(counts) == 1
        assert counts[0] >= 0
