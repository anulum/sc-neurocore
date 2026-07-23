# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBioAuditLog from former test_experiment.py

"""Focused suite: TestBioAuditLog from former test_experiment.py."""

from __future__ import annotations

from tests.test_bioware.experiment_support import *  # noqa: F403

class TestBioAuditLog:
    def test_log_entry(self) -> None:
        log = BioAuditLog(experiment_id="EXP001")
        log.log(BioAuditEntry(1, "2026-04-16T08:00:00", 100, 5, 500.0, 0.95))
        assert log.total_rounds == 1

    def test_to_list(self) -> None:
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        entries = log.to_list()
        assert entries[0]["round"] == 1
        assert entries[0]["spikes"] == 50

    def test_checksum_deterministic(self) -> None:
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        c1 = log.checksum()
        c2 = log.checksum()
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_checksum_changes(self) -> None:
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        c1 = log.checksum()
        log.log(BioAuditEntry(2, "2026-04-16", 60, 4, 400.0, 0.8))
        c2 = log.checksum()
        assert c1 != c2

    def test_checksum_uses_canonical_schema_and_experiment_identity(self) -> None:
        log = BioAuditLog(experiment_id="EXP001")
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        payload = {
            "schema": "sc-neurocore.bioware-audit.v1",
            "experiment_id": "EXP001",
            "entries": log.to_list(),
        }
        expected = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        assert log.checksum() == expected
