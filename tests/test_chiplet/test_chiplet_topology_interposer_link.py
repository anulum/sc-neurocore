# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestInterposerLink from former test_chiplet_topology.py

"""Focused suite: TestInterposerLink from former test_chiplet_topology.py."""

from __future__ import annotations

from chiplet_topology_support import *  # noqa: F403


class TestInterposerLink:
    """Interposer presets and fail-closed numerical contracts."""

    @pytest.mark.parametrize("technology", list(InterposerTech))
    def test_all_presets_are_physical(self, technology: InterposerTech) -> None:
        link = InterposerLink.from_tech(0, 1, technology)
        assert link.latency_ns >= 0
        assert link.bandwidth_gbps > 0
        assert 0 <= link.bit_error_rate <= 1

    def test_cowos_is_faster_than_organic(self) -> None:
        cowos = InterposerLink.from_tech(0, 1, InterposerTech.COWOS)
        organic = InterposerLink.from_tech(0, 1, InterposerTech.ORGANIC)
        assert cowos.latency_ns < organic.latency_ns
        assert cowos.bandwidth_gbps > organic.bandwidth_gbps

    def test_latency_and_jitter_determine_cycle_and_fifo_bounds(self) -> None:
        link = InterposerLink(src_die=0, dst_die=1, latency_ns=10.0, jitter_ns=5.0)
        assert link.latency_cycles == 2
        assert link.fifo_depth_log2 >= 3

    @pytest.mark.parametrize(
        ("constructor", "message"),
        [
            (lambda: InterposerLink(-1, 1), "src_die"),
            (lambda: InterposerLink(0, 1, latency_ns=math.nan), "latency_ns"),
            (lambda: InterposerLink(0, 1, jitter_ns=-1.0), "jitter_ns"),
            (lambda: InterposerLink(0, 1, bandwidth_gbps=0.0), "bandwidth_gbps"),
            (lambda: InterposerLink(0, 1, bit_error_rate=1.1), "bit_error_rate"),
            (lambda: InterposerLink(0, 1, data_width=0), "data_width"),
            (
                lambda: InterposerLink(0, 1, thermal_resistance_k_per_w=0.0),
                "thermal_resistance",
            ),
        ],
    )
    def test_invalid_link_contracts_fail(
        self, constructor: Callable[[], object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            constructor()
