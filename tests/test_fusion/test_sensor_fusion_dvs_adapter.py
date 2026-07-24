# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDVSAdapter from former test_sensor_fusion.py

"""Focused suite: TestDVSAdapter from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403


class TestDVSAdapter:
    def test_encode_events(self):
        ts = np.arange(10, dtype=np.float64) * 1000
        x = np.arange(10) % 128
        y = np.arange(10) % 128
        pol = np.ones(10, dtype=np.int8)
        stream = DVSAdapter.encode_events(ts, x, y, pol)
        assert stream.modality == SensorModality.DVS
        assert stream.num_events == 10
        assert "resolution" in stream.metadata

    def test_address_encoding(self):
        ts = np.array([0.0, 1000.0])
        x = np.array([5, 10])
        y = np.array([3, 7])
        pol = np.array([1, -1], dtype=np.int8)
        stream = DVSAdapter.encode_events(ts, x, y, pol, resolution=(128, 128))
        assert stream.addresses[0] == (3 * 128 + 5) % (128 * 128)
