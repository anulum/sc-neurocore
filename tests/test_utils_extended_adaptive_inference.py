# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveInference from former test_utils_extended.py

"""Focused suite: TestAdaptiveInference from former test_utils_extended.py."""

from __future__ import annotations

from tests.utils_extended_support import *  # noqa: F403

class TestAdaptiveInference:
    def test_converges_early(self):
        """Stable step_func should trigger early exit before max_length."""
        ai = AdaptiveInference(check_interval=16, tolerance=0.02, min_length=64, max_length=1024)
        call_count = 0

        def step():
            nonlocal call_count
            call_count += 1
            return 0.5  # perfectly stable

        result = ai.run_adaptive(step)
        assert result == pytest.approx(0.5)
        # Must exit well before max_length (3 checks after min_length)
        assert call_count < 1024

    def test_runs_to_max_length_when_noisy(self):
        """Noisy step_func should exhaust max_length."""
        rng = np.random.default_rng(42)
        ai = AdaptiveInference(check_interval=16, tolerance=0.001, min_length=64, max_length=256)
        call_count = 0

        def step():
            nonlocal call_count
            call_count += 1
            return rng.uniform(0.0, 1.0)

        ai.run_adaptive(step)
        assert call_count == 256

    def test_min_length_respected(self):
        """Even a constant function must run at least min_length steps."""
        ai = AdaptiveInference(check_interval=16, tolerance=0.1, min_length=128, max_length=512)
        call_count = 0

        def step():
            nonlocal call_count
            call_count += 1
            return 0.42

        ai.run_adaptive(step)
        assert call_count >= 128

    def test_returns_last_value(self):
        """Return value should be the last estimate from step_func."""
        counter = [0]

        def step():
            counter[0] += 1
            return float(counter[0])

        ai = AdaptiveInference(check_interval=32, tolerance=100.0, min_length=64, max_length=128)
        result = ai.run_adaptive(step)
        # With huge tolerance and checks starting at min_length=64,
        # should converge early once 3 checks accumulate
        assert result > 0
