# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUserProfile from former test_adaptive_audio.py

"""Focused suite: TestUserProfile from former test_adaptive_audio.py."""

from __future__ import annotations

from tests.adaptive_audio_support import *  # noqa: F403


class TestUserProfile(unittest.TestCase):
    def test_init(self) -> None:
        p = UserProfile()
        self.assertEqual(p.chronotype, Chronotype.BEAR)

    def test_chronotypes(self) -> None:
        for ct in Chronotype:
            p = UserProfile(chronotype=ct)
            self.assertEqual(p.chronotype, ct)

    def test_get_best_target(self) -> None:
        p = UserProfile()
        hz = p.get_best_target_hz()
        self.assertGreater(hz, 0)

    def test_update_from_session(self) -> None:
        p = UserProfile()
        # update_from_session(avg_evs, peak_evs, best_target_hz=None, band_powers=None)
        p.update_from_session(avg_evs=65.0, peak_evs=80.0)
        self.assertEqual(p.session_count, 1)

    def test_to_dict(self) -> None:
        d = UserProfile().to_dict()
        self.assertIn("chronotype", d)
        self.assertIn("user_id", d)

    def test_from_dict(self) -> None:
        p = UserProfile(chronotype=Chronotype.WOLF, user_id="test")
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        self.assertEqual(p2.chronotype, Chronotype.WOLF)
        self.assertEqual(p2.user_id, "test")

    def test_preferred_cost_weights(self) -> None:
        for ct in Chronotype:
            p = UserProfile(chronotype=ct)
            w = p.preferred_cost_weights
            self.assertIn("w_micro", w)
            self.assertGreater(w["w_micro"], 0)

    def test_avg_session_evs_after_update(self) -> None:
        p = UserProfile()
        p.update_from_session(avg_evs=65.0, peak_evs=80.0)
        self.assertEqual(p.session_count, 1)
