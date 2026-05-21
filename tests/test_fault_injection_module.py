# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault Injection Module Contract Tests

from __future__ import annotations

import pytest

from sc_neurocore.fault_injection.fault_injection import FaultInjectionResult, RadiationProfile


class TestRadiationProfileContracts:
    def test_presets_construct_valid_profiles(self):
        for profile in (
            RadiationProfile.terrestrial(),
            RadiationProfile.leo(),
            RadiationProfile.geo(),
            RadiationProfile.deep_space(),
        ):
            assert profile.name
            assert 0.0 <= profile.ber <= 1.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"name": ""}, "name"),
            ({"ber": -1e-6}, "ber"),
            ({"ber": 1.01}, "ber"),
            ({"ber": float("nan")}, "ber"),
            ({"description": 1}, "description"),
        ],
    )
    def test_rejects_invalid_contracts(self, kwargs, match):
        values = {
            "name": "LEO",
            "ber": 1e-7,
            "description": "ok",
        }
        values.update(kwargs)
        with pytest.raises(ValueError, match=match):
            RadiationProfile(**values)


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
