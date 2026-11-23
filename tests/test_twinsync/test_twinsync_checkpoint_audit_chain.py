# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCheckpointAuditChain from former test_twinsync.py

"""Focused suite: TestCheckpointAuditChain from former test_twinsync.py."""

from __future__ import annotations

from sc_neurocore.digital_twin.twinsync import (
    Checkpoint,
    CheckpointAuditChain,
)


class TestCheckpointAuditChain:
    """The hash chain over saved checkpoints."""

    def test_append_and_verify(self) -> None:
        """Five appended checkpoints form a chain that verifies."""
        chain = CheckpointAuditChain()
        for i in range(5):
            cp = Checkpoint(i, i * 100, 0, lfsr_state=i)
            cp.compute_checksum()
            chain.append(cp)
        assert chain.length == 5
        assert chain.verify() is True

    def test_tamper_detected(self) -> None:
        """A tampered checksum fails verification."""
        chain = CheckpointAuditChain()
        cp = Checkpoint(0, 0, 0)
        cp.compute_checksum()
        chain.append(cp)
        chain.chain[0] = (0, "tampered", chain.chain[0][2])
        assert chain.verify() is False
