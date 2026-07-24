# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProtocolConstants from former test_comm_aer.py

"""Focused suite: TestProtocolConstants from former test_comm_aer.py."""

from __future__ import annotations

from tests.comm_aer_support import *  # noqa: F403


class TestProtocolConstants:
    def test_magic(self):
        assert MAGIC == 0xAE01

    def test_header_size(self):
        assert HEADER_SIZE == 8

    def test_event_size(self):
        assert EVENT_SIZE == 8

    def test_max_events_fits_mtu(self):
        assert MAX_EVENTS_PER_PACKET * EVENT_SIZE + HEADER_SIZE <= 1500
