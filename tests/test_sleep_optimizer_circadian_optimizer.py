# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCircadianOptimizer from former test_sleep_optimizer.py

"""Focused suite: TestCircadianOptimizer from former test_sleep_optimizer.py."""

from __future__ import annotations

from tests.sleep_optimizer_support import *  # noqa: F403

class TestCircadianOptimizer(unittest.TestCase):
    def test_all_chronotypes_work(self):
        for ct in Chronotype:
            opt = CircadianOptimizer(ct)
            self.assertIsNotNone(opt.get_profile())

    def test_bear_defaults(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        p = opt.get_profile()
        self.assertEqual(p.bedtime_hour, 23.0)
        self.assertEqual(p.wake_hour, 7.0)

    def test_sleep_window(self):
        w = CircadianOptimizer(Chronotype.BEAR).get_sleep_window()
        self.assertIsInstance(w, tuple)
        self.assertEqual(len(w), 2)

    def test_is_in_sleep_window(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        self.assertTrue(opt.is_in_sleep_window(23.5))
        self.assertTrue(opt.is_in_sleep_window(2.0))
        self.assertFalse(opt.is_in_sleep_window(12.0))

    def test_recommended_protocol(self):
        for ct in Chronotype:
            proto = CircadianOptimizer(ct).get_recommended_protocol()
            self.assertIn(proto, PROTOCOL_REGISTRY)

    def test_melatonin_level(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        level = opt.melatonin_level(23.0)
        self.assertGreaterEqual(level, 0.0)
        self.assertLessEqual(level, 1.0)

    def test_melatonin_daytime_low(self):
        opt = CircadianOptimizer(Chronotype.BEAR)
        self.assertLess(opt.melatonin_level(14.0), 0.3)

    def test_to_dict(self):
        d = CircadianOptimizer(Chronotype.LION).to_dict()
        self.assertEqual(d["chronotype"], "lion")

    def test_different_protocols_for_different_types(self):
        wolf = CircadianOptimizer(Chronotype.WOLF).get_recommended_protocol()
        lion = CircadianOptimizer(Chronotype.LION).get_recommended_protocol()
        self.assertNotEqual(wolf, lion)
