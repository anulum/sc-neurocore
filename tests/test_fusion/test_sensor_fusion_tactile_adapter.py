# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTactileAdapter from former test_sensor_fusion.py

"""Focused suite: TestTactileAdapter from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403


class TestTactileAdapter:
    def test_encode_pressure(self):
        ts = np.arange(4, dtype=np.float64) * 100
        taxels = np.array([0, 1, 2, 3])
        pressures = np.array([0.5, 0.05, 0.8, 0.01])
        stream = TactileAdapter.encode_pressure(ts, taxels, pressures, threshold=0.1)
        assert stream.modality == SensorModality.TACTILE
        assert stream.num_events == 4
        assert stream.polarities[0] == 1  # above threshold
        assert stream.polarities[1] == -1  # below threshold
