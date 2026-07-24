# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpintronicDeviceConfig from former test_spintronic_mapper.py

"""Focused suite: TestSpintronicDeviceConfig from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403


class TestSpintronicDeviceConfig:
    def test_all_techs(self):
        for tech in SpintronicTech:
            cfg = SpintronicDeviceConfig.from_tech(tech)
            assert cfg.width_nm > 0
            assert cfg.switching_current_ua > 0

    def test_area(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        assert cfg.area_nm2 == cfg.width_nm * cfg.length_nm

    def test_switching_energy(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        assert cfg.switching_energy_fj > 0

    def test_switching_energy_uses_device_write_resistance(self):
        low_r = SpintronicDeviceConfig(
            switching_current_ua=40.0,
            switching_time_ns=2.0,
            write_resistance_ohm=2_000.0,
        )
        high_r = SpintronicDeviceConfig(
            switching_current_ua=40.0,
            switching_time_ns=2.0,
            write_resistance_ohm=8_000.0,
        )
        assert high_r.switching_energy_fj == 4.0 * low_r.switching_energy_fj

    def test_skyrmion_has_dmi(self):
        cfg = SpintronicDeviceConfig.from_tech(SpintronicTech.SKYRMION)
        assert cfg.material.dmi_strength_j_m2 > 0

    def test_sot_faster_than_stt(self):
        sot = SpintronicDeviceConfig.from_tech(SpintronicTech.SOT_MRAM)
        stt = SpintronicDeviceConfig.from_tech(SpintronicTech.STT_MTJ)
        assert sot.switching_time_ns < stt.switching_time_ns

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"width_nm": 0.0}, "width_nm must be positive"),
            ({"length_nm": 0.0}, "length_nm must be positive"),
            ({"thickness_nm": 0.0}, "thickness_nm must be positive"),
            ({"switching_current_ua": 0.0}, "switching_current_ua must be positive"),
            ({"switching_time_ns": 0.0}, "switching_time_ns must be positive"),
            ({"write_resistance_ohm": 0.0}, "write_resistance_ohm must be positive"),
            ({"parallel_resistance_ohm": 0.0}, "parallel_resistance_ohm must be positive"),
            ({"tmr_ratio": -0.1}, "tmr_ratio must be non-negative"),
        ],
    )
    def test_rejects_each_invalid_field(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SpintronicDeviceConfig(**kwargs)
