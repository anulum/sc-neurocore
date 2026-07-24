# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultModel from former test_resilience.py

"""Focused suite: TestFaultModel from former test_resilience.py."""

from __future__ import annotations

from tests.resilience_support import *  # noqa: F403


class TestFaultModel:
    def test_fields(self):
        fm = FaultModel(fault_type=FaultType.STUCK_AT_ZERO, rate=0.1)
        assert fm.rate == 0.1
        assert fm.layer_index is None

    def test_all_fault_types_exist(self):
        expected = {
            "stuck_at_0",
            "stuck_at_1",
            "weight_bit_flip",
            "dead_synapse",
            "noisy_membrane",
            "bitstream_bias",
        }
        actual = {ft.value for ft in FaultType}
        assert actual == expected
