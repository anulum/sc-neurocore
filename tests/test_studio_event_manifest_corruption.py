# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event manifest corruption at the public ledger boundary

"""Refuse damaged stored event contracts through actual SQLite and ledger reads."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_rows import StudioJobLedgerCorrupt
from tests.test_studio_training_config_storage import _configuration, _create


@pytest.mark.parametrize(
    "damage, explanation",
    [
        ("missing", "absent"),
        ("empty", "invalid"),
        ("malformed", "invalid"),
        ("nonobject", "canonical object"),
        ("noncanonical", "canonical object"),
        ("mixed", "mixes inline"),
        ("orphan", "no configuration reference"),
        ("binary", "not text"),
        ("no_configuration", "no training configuration"),
        ("wrong_kind", "non-training"),
        ("invalid_contract", "invalid"),
    ],
)
def test_damaged_manifest_cannot_be_read_as_a_valid_job(
    tmp_path: Path, damage: str, explanation: str
) -> None:
    """Even digest-consistent malformed bytes cannot produce a public training record."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    try:
        _create(ledger, _configuration(tmp_path / "recordings"))
        row = ledger.connection().execute("SELECT * FROM jobs WHERE job_id='job'").fetchone()
        config = json.loads(row["training_config"])
        original: str = row["training_event_data"]
        event_data: str | bytes | None = original
        kind = "training"
        if damage == "missing":
            event_data = None
        elif damage == "empty":
            event_data = ""
        elif damage == "malformed":
            event_data = "{"
        elif damage == "nonobject":
            event_data = "[]"
        elif damage == "noncanonical":
            event_data = original + "\n"
        elif damage == "mixed":
            config["event_data"] = {}
        elif damage == "orphan":
            del config["event_data_reference"]
        elif damage == "binary":
            event_data = original.encode()
        elif damage == "wrong_kind":
            kind = "analysis"
        elif damage == "invalid_contract":
            event_data = "{}"
        if damage in {"empty", "malformed", "nonobject", "noncanonical", "invalid_contract"}:
            assert isinstance(event_data, str)
            raw = event_data.encode()
            config["event_data_reference"]["bytes"] = len(raw)
            config["event_data_reference"]["sha256"] = hashlib.sha256(raw).hexdigest()
        compact = (
            None
            if damage == "no_configuration"
            else json.dumps(config, sort_keys=True, separators=(",", ":"))
        )
        # Simulate damaged storage after admission; the ordinary immutable guard is tested separately.
        ledger.connection().execute("DROP TRIGGER training_event_data_no_update")
        ledger.connection().execute(
            "UPDATE jobs SET training_config=?,training_event_data=?,kind=? WHERE job_id='job'",
            (compact, event_data, kind),
        )
        with pytest.raises(StudioJobLedgerCorrupt, match=explanation):
            ledger.record("job")
        with pytest.raises(StudioJobLedgerCorrupt, match=explanation):
            ledger.list_records()
    finally:
        ledger.close()


def test_sqlite_refuses_event_content_above_the_published_custody_bound(tmp_path: Path) -> None:
    """The on-disk byte constraint refuses oversized content even without its update guard."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    try:
        config = _configuration(tmp_path / "recordings")
        _create(ledger, config)
        ledger.connection().execute("DROP TRIGGER training_event_data_no_update")
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            ledger.connection().execute(
                "UPDATE jobs SET training_event_data=CAST(zeroblob(67108865) AS TEXT) "
                "WHERE job_id='job'"
            )
        assert ledger.record("job").training_config == config
    finally:
        ledger.close()
