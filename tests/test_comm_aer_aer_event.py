# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAEREvent from former test_comm_aer.py

"""Focused suite: TestAEREvent from former test_comm_aer.py."""

from __future__ import annotations

from tests.comm_aer_support import *  # noqa: F403

class TestAEREvent:
    def test_default_data(self):
        e = AEREvent(timestamp=100, neuron_id=5)
        assert e.data == 0

    def test_fields(self):
        e = AEREvent(timestamp=42, neuron_id=7, data=255)
        assert e.timestamp == 42
        assert e.neuron_id == 7
        assert e.data == 255
