# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFPGAFeedbackController from former test_bci_studio.py

"""Focused suite: TestFPGAFeedbackController from former test_bci_studio.py."""

from __future__ import annotations

from bci_studio_support import *  # noqa: F403

class TestFPGAFeedbackController(unittest.TestCase):
    def setUp(self):
        self.ctrl = FPGAFeedbackController()

    def test_serialize_deserialize(self):
        packet = self.ctrl.serialize(command=1, channel=42, amplitude=0.75, timestamp_us=12345.0)
        result = self.ctrl.deserialize(packet)
        self.assertEqual(result["command"], 1)
        self.assertEqual(result["channel"], 42)
        self.assertAlmostEqual(result["amplitude"], 0.75, places=4)
        self.assertAlmostEqual(result["timestamp_us"], 12345.0, places=1)

    def test_packet_size(self):
        packet = self.ctrl.serialize(command=0)
        self.assertEqual(len(packet), 16)  # DMA-aligned
