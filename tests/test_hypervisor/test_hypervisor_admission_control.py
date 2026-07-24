# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdmissionControl from former test_hypervisor.py

"""Focused suite: TestAdmissionControl from former test_hypervisor.py."""

from __future__ import annotations

from hypervisor_support import *  # noqa: F403


class TestAdmissionControl:
    def test_admit_ok(self):
        t = _tenant("new")
        t.qos.max_neurons = 512
        regions = {0: _region(0, neurons=1024)}
        ok, msg = admission_check(t, regions, {})
        assert ok is True

    def test_reject_insufficient(self):
        t = _tenant("new")
        t.qos.max_neurons = 2048
        regions = {0: _region(0, neurons=512)}
        ok, msg = admission_check(t, regions, {})
        assert ok is False
        assert "insufficient" in msg

    def test_reject_no_single_region_large_enough(self):
        # Aggregate free capacity is sufficient, but no single region can hold
        # the tenant: admission is refused because a tenant cannot be split.
        t = _tenant("new")
        t.qos.max_neurons = 800
        regions = {
            0: _region(0, neurons=512, base=0x4000_0000),
            1: _region(1, neurons=512, base=0x5000_0000),
        }
        ok, msg = admission_check(t, regions, {})
        assert ok is False
        assert msg == "no_single_region_large_enough"
