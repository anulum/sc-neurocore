# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFPGAFeedbackController from former test_bci_studio.py

"""Focused suite: TestFPGAFeedbackController from former test_bci_studio.py."""

from __future__ import annotations

import unittest
from sc_neurocore.bci_studio.bci_studio import (
    FPGAFeedbackController,
)


class TestFPGAFeedbackController(unittest.TestCase):
    """Packet encoding of the FPGA feedback controller."""

    def setUp(self) -> None:
        """Create a feedback controller."""
        self.ctrl = FPGAFeedbackController()

    def test_serialize_deserialize(self) -> None:
        """A packet decodes to the command, channel, amplitude and timestamp it encodes."""
        packet = self.ctrl.serialize(command=1, channel=42, amplitude=0.75, timestamp_us=12345.0)
        result = self.ctrl.deserialize(packet)
        self.assertEqual(result["command"], 1)
        self.assertEqual(result["channel"], 42)
        self.assertAlmostEqual(result["amplitude"], 0.75, places=4)
        self.assertAlmostEqual(result["timestamp_us"], 12345.0, places=1)

    def test_packet_size(self) -> None:
        """A packet is sixteen bytes long, aligned for DMA."""
        packet = self.ctrl.serialize(command=0)
        self.assertEqual(len(packet), 16)  # DMA-aligned
