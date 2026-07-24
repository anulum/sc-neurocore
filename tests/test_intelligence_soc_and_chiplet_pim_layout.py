# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPIMLayout from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestPIMLayout from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403


class TestPIMLayout:
    """Processing-in-Memory data layout planning."""

    def test_basic_layout(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        layout = plan_pim_layout(1000, 10000)
        assert layout.bank_count >= 1
        assert layout.neurons_per_bank >= 1
        assert layout.weights_per_bank >= 1
        assert 0 < layout.bank_utilisation <= 1.0
        assert layout.parallel_factor >= 1

    def test_layout_map_regions(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        layout = plan_pim_layout(1000, 50000, num_banks=16)
        assert "neuron_state" in layout.layout_map
        assert "synaptic_weights" in layout.layout_map

    def test_large_network_uses_more_banks(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        small = plan_pim_layout(100, 1000, num_banks=16)
        large = plan_pim_layout(100000, 10000000, num_banks=16)
        assert large.bank_count >= small.bank_count

    def test_respects_bank_limit(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        layout = plan_pim_layout(1000000, 100000000, num_banks=8)
        assert layout.bank_count <= 8

    def test_custom_bank_size(self):
        from sc_neurocore.compiler.intelligence import plan_pim_layout

        layout = plan_pim_layout(100, 1000, bank_size_kb=32)
        assert layout.bank_count >= 1
