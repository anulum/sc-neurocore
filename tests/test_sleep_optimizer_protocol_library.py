# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProtocolLibrary from former test_sleep_optimizer.py

"""Focused suite: TestProtocolLibrary from former test_sleep_optimizer.py."""

from __future__ import annotations

import unittest
from sc_neurocore.sleep import (
    get_protocol,
    list_protocols,
    SleepStage,
)
from sc_neurocore.sleep.protocol_library import (
    PROTOCOL_REGISTRY,
    StageAudioParams,
)


class TestProtocolLibrary(unittest.TestCase):
    """The sleep protocol library: its six protocols, stage parameters and targets."""

    def test_six_protocols(self) -> None:
        """The registry holds six protocols."""
        self.assertEqual(len(PROTOCOL_REGISTRY), 6)

    def test_get_valid(self) -> None:
        """A known protocol name returns that protocol."""
        self.assertEqual(get_protocol("insomnia_relief").name, "insomnia_relief")

    def test_get_invalid(self) -> None:
        """An unknown protocol name is refused."""
        with self.assertRaises((ValueError, KeyError)):
            get_protocol("nonexistent")

    def test_list(self) -> None:
        """Listing returns all six protocols."""
        protos = list_protocols()
        self.assertEqual(len(protos), 6)

    def test_stage_params(self) -> None:
        """A protocol gives stage audio parameters."""
        for name, proto in PROTOCOL_REGISTRY.items():
            for stage in SleepStage:
                audio = proto.get_audio_for_stage(stage)
                self.assertIsInstance(audio, StageAudioParams)

    def test_targets_sum_one(self) -> None:
        """Each protocol's stage targets sum to one."""
        for name, proto in PROTOCOL_REGISTRY.items():
            total = sum(proto.stage_targets.values())
            self.assertAlmostEqual(total, 1.0, places=2, msg=f"{name} targets sum={total}")

    def test_target_stage_progression(self) -> None:
        """Target stage progression."""
        proto = get_protocol("insomnia_relief")
        # get_target_stage takes a single float progress in [0, 1]
        early = proto.get_target_stage(0.01)
        self.assertIn(early, list(SleepStage))
        mid = proto.get_target_stage(0.5)
        self.assertIn(mid, list(SleepStage))

    def test_to_dict(self) -> None:
        """The dictionary form names the protocol."""
        d = get_protocol("power_nap").to_dict()
        self.assertEqual(d["name"], "power_nap")

    def test_power_nap_short(self) -> None:
        """The power nap lasts at most 30 minutes."""
        self.assertLessEqual(get_protocol("power_nap").total_duration_min, 30.0)
