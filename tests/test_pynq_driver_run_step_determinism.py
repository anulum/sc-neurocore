# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRunStepDeterminism from former test_pynq_driver.py

"""Focused suite: TestRunStepDeterminism from former test_pynq_driver.py."""

from __future__ import annotations

from tests.pynq_driver_support import *  # noqa: F403


class TestRunStepDeterminism:
    """Two drivers built with the same seed produce identical run_step output."""

    def test_run_step_same_seed_identical_first_call(self):
        a = SC_NeuroCore_Driver(mode="EMULATION", seed=123)
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=123)
        np.testing.assert_array_equal(a.run_step(None), b.run_step(None))

    def test_run_step_same_seed_identical_sequence(self):
        a = SC_NeuroCore_Driver(mode="EMULATION", seed=99)
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=99)
        for _ in range(50):
            np.testing.assert_array_equal(a.run_step(None), b.run_step(None))

    def test_run_step_different_seeds_differ(self):
        a = SC_NeuroCore_Driver(mode="EMULATION", seed=1)
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=2)
        out_a = a.run_step(None)
        out_b = b.run_step(None)
        # Two distinct seeds: shape matches, values differ
        assert out_a.shape == out_b.shape == (16,)
        assert not np.array_equal(out_a, out_b)

    def test_run_step_global_numpy_seed_does_not_leak(self):
        """np.random.seed(...) between constructions must not affect output."""
        np.random.seed(0)
        a = SC_NeuroCore_Driver(mode="EMULATION", seed=42)
        out_a = a.run_step(None)

        np.random.seed(99999)
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=42)
        out_b = b.run_step(None)
        np.testing.assert_array_equal(out_a, out_b)

    def test_run_step_default_seed_is_42(self):
        a = SC_NeuroCore_Driver(mode="EMULATION")
        b = SC_NeuroCore_Driver(mode="EMULATION", seed=42)
        np.testing.assert_array_equal(a.run_step(None), b.run_step(None))
