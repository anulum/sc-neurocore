# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAuditValidation from former test_validation_experiment.py

"""Focused suite: TestAuditValidation from former test_validation_experiment.py."""

from __future__ import annotations

from tests.test_bioware.validation_experiment_support import *  # noqa: F403

class TestAuditValidation:
    def test_entry_rejects_invalid_metadata(self) -> None:
        with pytest.raises(ValueError, match="timestamp_iso must not be empty"):
            BioAuditEntry(1, " ", 0, 0, 0.0, 0.0)
        with pytest.raises(ValueError, match="ISO-8601"):
            BioAuditEntry(1, "not-a-time", 0, 0, 0.0, 0.0)
        with pytest.raises(ValueError, match="health_score must be <= 1"):
            BioAuditEntry(1, "2026-07-13", 0, 0, 0.0, 1.1)
        with pytest.raises(TypeError, match="notes must be a string"):
            BioAuditEntry(1, "2026-07-13", 0, 0, 0.0, 0.0, notes=cast(Any, 1))

    def test_log_rejects_invalid_identity_entries_and_order(self) -> None:
        with pytest.raises(TypeError, match="experiment_id must be a string"):
            BioAuditLog(experiment_id=cast(Any, 1))
        with pytest.raises(ValueError, match="whitespace"):
            BioAuditLog(experiment_id=" ")
        with pytest.raises(TypeError, match="BioAuditEntry"):
            BioAuditLog(entries=cast(Any, [object()]))
        with pytest.raises(ValueError, match="increase strictly"):
            BioAuditLog(entries=[_entry(2), _entry(1)])
        log = BioAuditLog(entries=[_entry(1)])
        with pytest.raises(TypeError, match="BioAuditEntry"):
            log.log(cast(Any, object()))
        with pytest.raises(ValueError, match="increase strictly"):
            log.log(_entry(1))
