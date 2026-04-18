# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Full coverage tests for identity/director.py

"""Exercise every branch of DirectorController.diagnose(), correct(),
report(), and all helper functions."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.identity.director import (
    DirectorController,
    _add_weight_noise,
    _homeostatic_scale,
    _prune_weak,
    _grow_synapses,
)


def _make_substrate():
    sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
    sub.run(duration=0.1, dt=0.001)
    return sub


class TestDiagnoseRateTooHigh:
    def test_rate_too_high(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 50.0,
                "cv": 1.0,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "rate_too_high" in problems


class TestDiagnoseRateTooLow:
    def test_rate_too_low(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 2.0,
                "cv": 1.0,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "rate_too_low" in problems


class TestDiagnoseSilent:
    def test_silent(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 0.0,
                "cv": float("nan"),
                "fano": float("nan"),
                "perm_entropy": float("nan"),
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "silent" in problems


class TestDiagnoseTooRegular:
    def test_too_regular(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 10.0,
                "cv": 0.1,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "too_regular" in problems


class TestDiagnoseTooChaoticBursty:
    def test_too_chaotic(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 10.0,
                "cv": 3.0,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "too_chaotic" in problems

    def test_bursty(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 10.0,
                "cv": 1.0,
                "fano": 5.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "bursty" in problems


class TestDiagnoseConnectivity:
    def test_connectivity_too_dense(self):
        sub = _make_substrate()
        # Force dense connectivity
        sub.proj_ee.data[:] = 1.0
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 10.0,
                "cv": 1.0,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "connectivity_too_dense" in problems

    def test_connectivity_too_sparse(self):
        sub = _make_substrate()
        # Force sparse connectivity
        sub.proj_ee.data[:] = 0.0
        sub.proj_ee.data[0] = 0.01  # keep one nonzero so density < 0.05
        director = DirectorController(sub)
        with patch.object(
            director,
            "monitor",
            return_value={
                "mean_rate": 10.0,
                "cv": 1.0,
                "fano": 1.0,
                "perm_entropy": 0.8,
                "n_steps": 200,
            },
        ):
            problems = director.diagnose()
            assert "connectivity_too_sparse" in problems


class TestCorrect:
    def test_correct_rate_too_high(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        ie_before = sub.proj_ie.data.copy()
        with patch.object(director, "diagnose", return_value=["rate_too_high"]):
            director.correct()
        # proj_ie *= 1.1 — inhibitory weights are negative, so |w| increases
        assert np.mean(np.abs(sub.proj_ie.data)) > np.mean(np.abs(ie_before)) - 1e-6

    def test_correct_rate_too_low(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        ie_before = sub.proj_ie.data.copy()
        with patch.object(director, "diagnose", return_value=["rate_too_low"]):
            director.correct()
        # proj_ie *= 0.9 — inhibitory weights |w| decreases
        assert np.mean(np.abs(sub.proj_ie.data)) < np.mean(np.abs(ie_before)) + 1e-6

    def test_correct_silent(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["silent"]):
            director.correct()

    def test_correct_too_regular(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        ee_before = sub.proj_ee.data.copy()
        with patch.object(director, "diagnose", return_value=["too_regular"]):
            director.correct()
        # Weights should change (noise added)
        assert not np.allclose(sub.proj_ee.data, ee_before)

    def test_correct_too_chaotic(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["too_chaotic"]):
            director.correct()

    def test_correct_bursty(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["bursty"]):
            director.correct()

    def test_correct_connectivity_too_dense(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["connectivity_too_dense"]):
            director.correct()

    def test_correct_connectivity_too_sparse(self):
        sub = _make_substrate()
        sub.proj_ee.data[:] = 0.0  # all zero
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["connectivity_too_sparse"]):
            director.correct()
        # Some weights should be grown
        assert np.any(sub.proj_ee.data > 0)

    def test_correct_healthy_no_action(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        ee_before = sub.proj_ee.data.copy()
        with patch.object(director, "diagnose", return_value=[]):
            director.correct()
        np.testing.assert_array_equal(sub.proj_ee.data, ee_before)

    def test_corrections_counter(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        assert director._corrections_applied == 0
        with patch.object(director, "diagnose", return_value=["rate_too_high"]):
            director.correct()
        assert director._corrections_applied == 1


class TestReport:
    def test_report_healthy(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=[]):
            report = director.report()
        assert "healthy" in report

    def test_report_problems(self):
        sub = _make_substrate()
        director = DirectorController(sub)
        with patch.object(director, "diagnose", return_value=["rate_too_high", "bursty"]):
            report = director.report()
        assert "rate_too_high" in report
        assert "bursty" in report


class TestHelperFunctions:
    def test_add_weight_noise(self):
        data = np.array([0.0, 0.5, 0.3, 0.0, 0.8])
        original = data.copy()
        _add_weight_noise(data, scale=0.1)
        # Nonzero weights should change
        assert not np.allclose(data[1:3], original[1:3])
        # Weights should be non-negative
        assert np.all(data >= 0)
        # Zero weights should stay zero (no noise applied to zeros)
        assert data[0] == 0.0

    def test_homeostatic_scale(self):
        data = np.array([0.0, 0.1, 0.5, 0.9, 0.0])
        _homeostatic_scale(data, factor=0.5)
        # Weights should be pulled toward mean
        assert np.all(data >= 0)

    def test_homeostatic_scale_all_zero(self):
        data = np.zeros(5)
        _homeostatic_scale(data, factor=0.9)
        assert np.all(data == 0)

    def test_prune_weak(self):
        data = np.array([0.005, 0.5, 0.002, 0.8, 0.001])
        _prune_weak(data, threshold=0.01)
        assert data[0] == 0.0
        assert data[2] == 0.0
        assert data[4] == 0.0
        assert data[1] == 0.5
        assert data[3] == 0.8

    def test_grow_synapses(self):
        data = np.array([0.0, 0.5, 0.0, 0.0, 0.3])
        _grow_synapses(data, fraction=0.5, seed=42)
        # Some zeros should become positive
        grown = data[np.array([0, 2, 3])]
        assert np.any(grown > 0)

    def test_grow_synapses_no_zeros(self):
        data = np.array([0.1, 0.5, 0.3])
        _grow_synapses(data, fraction=0.5, seed=42)
        # No zeros to grow — should not crash
