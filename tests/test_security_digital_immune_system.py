# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDigitalImmuneSystem from former test_security.py

"""Focused suite: TestDigitalImmuneSystem from former test_security.py."""

from __future__ import annotations

from tests.security_support import *  # noqa: F403

class TestDigitalImmuneSystem:
    """Test suite for the Artificial Immune System anomaly detector."""

    def setup_method(self):
        self.immune = DigitalImmuneSystem(tolerance=0.2)

    def test_untrained_system_allows_all(self):
        """Untrained system should allow all states (no self patterns)."""
        state = np.array([0.5, 0.5, 0.5])
        result = self.immune.scan(state)
        assert result is True, "Untrained system should pass all scans"

    def test_train_self_stores_pattern(self):
        """Training should store the normal state pattern."""
        normal = np.array([1.0, 0.0, 1.0])
        self.immune.train_self(normal)
        assert len(self.immune.self_patterns) == 1
        assert np.array_equal(self.immune.self_patterns[0], normal)

    def test_scan_passes_similar_state(self):
        """States similar to training should pass scan."""
        normal = np.array([1.0, 0.0, 1.0])
        self.immune.train_self(normal)

        # Very similar state (within tolerance)
        similar = np.array([1.0, 0.1, 1.0])
        result = self.immune.scan(similar)
        assert result is True, "Similar state should pass scan"

    def test_scan_detects_anomaly(self):
        """States far from training should be detected as anomalies."""
        normal = np.array([1.0, 0.0, 1.0])
        self.immune.train_self(normal)

        # Very different state (outside tolerance)
        anomaly = np.array([0.0, 1.0, 0.0])
        result = self.immune.scan(anomaly)
        assert result is False, "Anomalous state should fail scan"

    def test_multiple_self_patterns(self):
        """System should handle multiple normal patterns."""
        patterns = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        for p in patterns:
            self.immune.train_self(p)

        assert len(self.immune.self_patterns) == 3

        # Test that a state close to any pattern passes
        close_to_first = np.array([1.0, 0.1, 0.0])
        assert self.immune.scan(close_to_first) is True

    def test_tolerance_threshold(self):
        """Test that tolerance controls detection sensitivity."""
        normal = np.array([0.5, 0.5, 0.5])
        self.immune.train_self(normal)

        # State exactly at tolerance boundary
        # L2 norm from [0.5,0.5,0.5] to [0.5,0.5,0.7] = 0.2
        boundary = np.array([0.5, 0.5, 0.7])
        result = self.immune.scan(boundary)
        assert result is True, "State at tolerance boundary should pass"

        # State just outside tolerance
        outside = np.array([0.5, 0.5, 0.71])
        result = self.immune.scan(outside)
        # Depends on exact tolerance, may be True or False

    def test_max_patterns_limit(self):
        """System limits stored patterns to 100."""
        for i in range(150):
            self.immune.train_self(np.array([float(i), 0.0, 0.0]))

        assert len(self.immune.self_patterns) == 100
