# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeviceCatalog from former test_hardware.py

"""Focused suite: TestDeviceCatalog from former test_hardware.py."""

from __future__ import annotations

from tests.hardware_support import *  # noqa: F403


class TestDeviceCatalog:
    def test_all_families_have_specs(self):
        for family in DeviceFamily:
            assert family in DEVICE_CATALOG, f"{family.name} missing from catalog"

    @pytest.mark.parametrize("family", list(DeviceFamily))
    def test_device_spec_valid(self, family):
        spec = DEVICE_CATALOG[family]
        assert spec.cores > 0
        assert spec.neurons_per_core > 0
        assert spec.precision_bits > 0
        assert spec.tick_ns > 0
        assert spec.power_per_core_mw >= 0

    def test_get_device_by_enum(self):
        spec = get_device(DeviceFamily.LOIHI)
        assert spec.family == DeviceFamily.LOIHI

    def test_get_device_by_string(self):
        spec = get_device("loihi2")
        assert spec.family == DeviceFamily.LOIHI2

    def test_get_device_unknown_raises(self):
        with pytest.raises((ValueError, KeyError)):
            get_device("nonexistent")

    def test_loihi_specs_match_datasheet(self):
        loihi = get_device(DeviceFamily.LOIHI)
        assert loihi.cores == 128
        assert loihi.neurons_per_core == 1024
        assert loihi.weight_bits == 9

    def test_spinnaker_specs(self):
        spin = get_device(DeviceFamily.SPINNAKER)
        assert spin.cores == 18
        assert spin.supports_learning is True
