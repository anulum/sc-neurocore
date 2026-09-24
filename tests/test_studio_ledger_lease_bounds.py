# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Durable lease duration contracts

"""Invalid lease configuration must fail before creating authoritative state."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_invalid_lease_refuses_before_directory_creation(tmp_path: Path, duration: float) -> None:
    """Invalid durations cannot leave a database or root for recovery to adopt."""
    root = tmp_path / "authority"
    with pytest.raises(ValueError, match="lease duration must"):
        ledger = StudioJobLedger(root=root, lease_seconds=duration)
        ledger.close()
    assert not root.exists()


@pytest.mark.parametrize("duration", [0.25, 1.0, 60.0])
def test_finite_lease_persists_and_renews_for_same_supervisor(
    tmp_path: Path, duration: float
) -> None:
    """Real creation and heartbeat retain configured duration and supervisor identity."""
    now = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
    ledger = StudioJobLedger(
        root=tmp_path / "authority",
        lease_seconds=duration,
        supervisor="lease-owner",
        clock=lambda: now,
    )
    try:
        record = ledger.create(
            job_id="sj_lease_bounds",
            kind="analysis",
            actor="studio",
            workspace="default",
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission=None,
            execution_model="thread",
        ).record
        expected = now + timedelta(seconds=duration)
        assert record.lease_expires_at_utc == expected.isoformat().replace("+00:00", "Z")
        assert record.lease_owner == "lease-owner"
        now += timedelta(seconds=2)
        ledger.heartbeat(record.job_id)
        observed = ledger.record(record.job_id)
        expected = now + timedelta(seconds=duration)
        assert observed.lease_expires_at_utc == expected.isoformat().replace("+00:00", "Z")
        assert observed.lease_owner == "lease-owner"
    finally:
        ledger.close()
