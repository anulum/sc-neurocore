# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProtocolConstants from former test_comm_aer.py

"""Focused suite: TestProtocolConstants from former test_comm_aer.py."""

from __future__ import annotations

from sc_neurocore.comm.aer_udp import (
    EVENT_SIZE,
    HEADER_SIZE,
    MAGIC,
    MAX_EVENTS_PER_PACKET,
)


class TestProtocolConstants:
    """The AER wire protocol constants."""

    def test_magic(self) -> None:
        """The packet magic is 0xAE01."""
        assert MAGIC == 0xAE01

    def test_header_size(self) -> None:
        """The header is eight bytes."""
        assert HEADER_SIZE == 8

    def test_event_size(self) -> None:
        """Each event is eight bytes."""
        assert EVENT_SIZE == 8

    def test_max_events_fits_mtu(self) -> None:
        """A full packet fits in the network MTU."""
        assert MAX_EVENTS_PER_PACKET * EVENT_SIZE + HEADER_SIZE <= 1500
