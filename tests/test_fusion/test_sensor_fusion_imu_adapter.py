# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIMUAdapter from former test_sensor_fusion.py

"""Focused suite: TestIMUAdapter from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403


class TestIMUAdapter:
    def test_encode_angular_rate(self):
        ts = np.arange(10, dtype=np.float64) * 100
        axes = np.zeros(10, dtype=np.int64)
        rates = np.array([10, 3, -20, 2, 15, -1, 30, 4, -8, 0], dtype=np.float64)
        stream = IMUAdapter.encode_angular_rate(ts, axes, rates, deadzone_dps=5.0)
        assert stream.modality == SensorModality.PROPRIOCEPTIVE
        assert stream.num_events < 10  # some filtered by deadzone

    def test_deadzone_filters_small(self):
        ts = np.arange(5, dtype=np.float64) * 100
        axes = np.zeros(5, dtype=np.int64)
        rates = np.array([1, 2, 3, 4, 100], dtype=np.float64)
        stream = IMUAdapter.encode_angular_rate(ts, axes, rates, deadzone_dps=50.0)
        assert stream.num_events == 1  # only 100 > 50
