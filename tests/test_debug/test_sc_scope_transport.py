# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTransport from former test_sc_scope.py

"""Focused suite: TestTransport from former test_sc_scope.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.debug.sc_scope import (
    TransportBackend,
    TransportConfig,
    TransportType,
)


class TestTransport:
    """The simulated transport backend."""

    def test_simulated_connect(self) -> None:
        """The simulated transport connects and disconnects."""
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        assert tb.connect() is True
        assert tb.is_connected is True
        tb.disconnect()
        assert tb.is_connected is False

    def test_simulated_read(self) -> None:
        """A read returns eight uint32 words."""
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        tb.connect()
        words = tb.read_bitstream(8, layer_id=0)
        assert words is not None
        assert len(words) == 8
        assert words.dtype == np.uint32

    def test_read_without_connect(self) -> None:
        """A read before connecting returns nothing."""
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        assert tb.read_bitstream(8) is None

    def test_bytes_counted(self) -> None:
        """Received bytes are counted."""
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        tb.connect()
        tb.read_bitstream(16)
        assert tb.bytes_received == 64

    def test_multiple_reads(self) -> None:
        """Two reads return different words."""
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        tb.connect()
        r1 = tb.read_bitstream(4)
        r2 = tb.read_bitstream(4)
        assert r1 is not None
        assert r2 is not None
        assert not np.array_equal(r1, r2)
