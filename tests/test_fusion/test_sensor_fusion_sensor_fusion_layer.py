# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSensorFusionLayer from former test_sensor_fusion.py

"""Focused suite: TestSensorFusionLayer from former test_sensor_fusion.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.fusion.sensor_fusion import (
    SensorFusionLayer,
    SensorModality,
)
from sensor_fusion_support import (
    _make_stream,
)


class TestSensorFusionLayer:
    """Fusing event streams of several modalities."""

    def test_fuse_two_streams(self) -> None:
        """Two streams fuse into one bitstream with their counts."""
        layer = SensorFusionLayer(num_channels=16, bitstream_length=128, seed=42)
        s1 = _make_stream(SensorModality.DVS, 50, seed=0)
        s2 = _make_stream(SensorModality.TACTILE, 50, seed=1)
        fused, metrics = layer.fuse([s1, s2])
        assert fused.shape == (16, 128)
        assert metrics.num_streams == 2
        assert metrics.total_events == 100

    def test_fuse_empty_list(self) -> None:
        """Fusing nothing gives an all-zero bitstream and no streams."""
        layer = SensorFusionLayer()
        fused, metrics = layer.fuse([])
        assert np.sum(fused) == 0
        assert metrics.num_streams == 0

    def test_fuse_single_stream(self) -> None:
        """One stream fuses into a bitstream of the layer's shape."""
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        s = _make_stream(SensorModality.COCHLEA, 30, seed=0)
        fused, metrics = layer.fuse([s])
        assert fused.shape == (8, 64)
        assert metrics.num_streams == 1

    def test_modality_weighting(self) -> None:
        """A low modality weight gives no more ones than full weight."""
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        layer.set_weight(SensorModality.DVS, 0.1)
        s = _make_stream(SensorModality.DVS, 100, seed=0)
        fused_weighted, _ = layer.fuse([s], use_attention=False)

        layer2 = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        fused_full, _ = layer2.fuse([s], use_attention=False)

        assert np.sum(fused_weighted) <= np.sum(fused_full)

    def test_latency_recorded(self) -> None:
        """Fusion records a positive latency."""
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        s = _make_stream(SensorModality.DVS, 50)
        _, metrics = layer.fuse([s])
        assert metrics.latency_us > 0.0

    def test_cross_modal_scc_bounded(self) -> None:
        """The cross-modal SCC lies between minus one and one."""
        layer = SensorFusionLayer(num_channels=8, bitstream_length=128, seed=42)
        s1 = _make_stream(SensorModality.DVS, 80, seed=0)
        s2 = _make_stream(SensorModality.TACTILE, 80, seed=1)
        _, metrics = layer.fuse([s1, s2])
        assert -1.0 <= metrics.cross_modal_scc <= 1.0

    def test_three_modality_fusion(self) -> None:
        """Three modalities fuse into one bitstream."""
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        streams = [
            _make_stream(SensorModality.DVS, 30, seed=0),
            _make_stream(SensorModality.TACTILE, 30, seed=1),
            _make_stream(SensorModality.COCHLEA, 30, seed=2),
        ]
        fused, metrics = layer.fuse(streams)
        assert metrics.num_streams == 3
        assert fused.shape == (8, 64)

    def test_fuse_without_attention(self) -> None:
        """Fusion without attention keeps the layer's shape."""
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        s1 = _make_stream(SensorModality.DVS, 50, seed=0)
        s2 = _make_stream(SensorModality.TACTILE, 50, seed=1)
        fused, metrics = layer.fuse([s1, s2], use_attention=False)
        assert fused.shape == (8, 64)
        assert metrics.fused_popcount >= 0
