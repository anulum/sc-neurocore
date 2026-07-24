# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultInjectionResultContracts from former test_fault_injection_module.py

"""Focused suite: TestFaultInjectionResultContracts from former test_fault_injection_module.py."""

from __future__ import annotations

from tests.fault_injection_module_support import *  # noqa: F403


class TestFaultInjectionResultContracts:
    def test_probability_properties_follow_popcount_ratio(self):
        result = FaultInjectionResult(
            original_popcount=6,
            corrupted_popcount=5,
            bits_flipped=1,
            bitstream_length=10,
        )
        assert result.probability_original == 0.6
        assert result.probability_corrupted == 0.5
        assert result.absolute_error == pytest.approx(0.1)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"original_popcount": -1}, "original_popcount"),
            ({"corrupted_popcount": -1}, "corrupted_popcount"),
            ({"bits_flipped": -1}, "bits_flipped"),
            ({"bitstream_length": -1}, "bitstream_length"),
            ({"original_popcount": 11}, "original_popcount"),
            ({"corrupted_popcount": 11}, "corrupted_popcount"),
            ({"bits_flipped": 11}, "bits_flipped"),
        ],
    )
    def test_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "original_popcount": 6,
            "corrupted_popcount": 5,
            "bits_flipped": 1,
            "bitstream_length": 10,
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            FaultInjectionResult(**values)
