# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHDCBinding from former test_sensor_fusion.py

"""Focused suite: TestHDCBinding from former test_sensor_fusion.py."""

from __future__ import annotations

from sensor_fusion_support import *  # noqa: F403


class TestHDCBinding:
    def test_hypervector_dimension(self):
        hdc = HDCBinding(dim=2048)
        hv = hdc.get_hypervector("dvs")
        assert len(hv) == 2048

    def test_hypervector_deterministic(self):
        hdc = HDCBinding(dim=512, seed=42)
        a = hdc.get_hypervector("test")
        b = hdc.get_hypervector("test")
        np.testing.assert_array_equal(a, b)

    def test_bind_is_self_inverse(self):
        hdc = HDCBinding(dim=1024, seed=42)
        a = hdc.get_hypervector("a")
        b = hdc.get_hypervector("b")
        bound = hdc.bind(a, b)
        unbound = hdc.bind(bound, b)
        np.testing.assert_array_equal(unbound, a)

    def test_bundle_majority_vote(self):
        hdc = HDCBinding(dim=1024, seed=42)
        a = np.ones(1024, dtype=np.uint8)
        b = np.ones(1024, dtype=np.uint8)
        c = np.zeros(1024, dtype=np.uint8)
        result = hdc.bundle([a, b, c])
        assert np.sum(result) == 1024  # majority is 1

    def test_bundle_empty_returns_zero_hypervector(self):
        hdc = HDCBinding(dim=64)
        result = hdc.bundle([])
        assert result.shape == (64,)
        assert np.sum(result) == 0

    def test_similarity_identical(self):
        hdc = HDCBinding(dim=512, seed=42)
        a = hdc.get_hypervector("x")
        assert hdc.similarity(a, a) == 1.0

    def test_similarity_random_near_half(self):
        hdc = HDCBinding(dim=4096, seed=42)
        a = hdc.get_hypervector("x")
        b = hdc.get_hypervector("y")
        sim = hdc.similarity(a, b)
        assert 0.3 < sim < 0.7

    def test_encode_stream(self):
        hdc = HDCBinding(dim=1024, seed=42)
        s = _make_stream(SensorModality.DVS, 50, seed=0)
        hv = hdc.encode_stream(s)
        assert len(hv) == 1024
        assert hv.dtype == np.uint8

    def test_different_modalities_different_encoding(self):
        hdc = HDCBinding(dim=1024, seed=42)
        s1 = _make_stream(SensorModality.DVS, 50, seed=0)
        s2 = _make_stream(SensorModality.COCHLEA, 50, seed=0)
        hv1 = hdc.encode_stream(s1)
        hv2 = hdc.encode_stream(s2)
        assert not np.array_equal(hv1, hv2)
