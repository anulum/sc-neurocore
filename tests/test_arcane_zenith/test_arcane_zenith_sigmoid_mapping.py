# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSigmoidMapping from former test_arcane_zenith.py

"""Focused suite: TestSigmoidMapping from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403


class TestSigmoidMapping:
    """``_map_to_range`` = sigmoid(10*(w-0.5)) interpolated into [min, max]."""

    @pytest.fixture
    def core(self) -> ArcaneZenithCognitiveCore:
        return create_arcane_neuron_with_zenith_plasticity(backend="torch")

    def test_endpoint_w_zero_approaches_min(self, core):
        # sigmoid(-5) ≈ 0.0067 → result ≈ min + 0.0067*(max-min)
        out = core._map_to_range(0.0, 10.0, 110.0)
        assert 10.0 <= out <= 11.0
        expected = 10.0 + (1.0 / (1.0 + math.exp(5.0))) * 100.0
        assert abs(out - expected) < 1e-6

    def test_endpoint_w_one_approaches_max(self, core):
        # sigmoid(+5) ≈ 0.9933 → result ≈ min + 0.9933*(max-min)
        out = core._map_to_range(1.0, 10.0, 110.0)
        assert 109.0 <= out <= 110.0
        expected = 10.0 + (1.0 / (1.0 + math.exp(-5.0))) * 100.0
        assert abs(out - expected) < 1e-6

    def test_midpoint_w_half_is_exact_midpoint(self, core):
        out = core._map_to_range(0.5, 10.0, 110.0)
        assert abs(out - 60.0) < 1e-9

    def test_strict_monotonic_in_weight(self, core):
        samples = np.linspace(0.0, 1.0, 41)
        mapped = [core._map_to_range(float(w), 0.0, 1.0) for w in samples]
        assert all(mapped[i + 1] > mapped[i] for i in range(len(mapped) - 1))

    def test_clamp_above_one_saturates_at_max(self, core):
        # Sigmoid saturates: extreme w still cannot exceed max by construction.
        assert core._map_to_range(5.0, 10.0, 20.0) == pytest.approx(20.0, abs=1e-3)

    def test_clamp_below_zero_saturates_at_min(self, core):
        assert core._map_to_range(-5.0, 10.0, 20.0) == pytest.approx(10.0, abs=1e-3)
