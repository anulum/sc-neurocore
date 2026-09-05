# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job recovery rules

"""Leases, supervisor identity, and what recovery concludes from them.

Recovery has exactly three answers about a job it finds alive: someone is still
running it, nobody is, or this host cannot tell. These cases pin each one,
including the case a lazier implementation would get wrong — a supervisor on
another host, which is unknown rather than dead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import (
    _process_start_token,
    supervisor_identity,
    supervisor_is_alive,
)

UTC_CLOCK_START = datetime.fromisoformat("2026-09-06T00:00:00+00:00")


def _ledger(root: Path, **kwargs: object) -> StudioJobLedger:
    return StudioJobLedger(root=root, **kwargs)  # type: ignore[arg-type]


def _admit(ledger: StudioJobLedger, job_id: str = "sj_0000000000000001", **kwargs: object):
    fields: dict[str, object] = {
        "job_id": job_id,
        "kind": "analysis",
        "actor": "alice",
        "workspace": "default",
        "request_id": None,
        "idempotency_key": None,
        "experiment_sha256": None,
        "admission": None,
        "execution_model": "thread",
    }
    fields.update(kwargs)
    return ledger.create(**fields)  # type: ignore[arg-type]


def _manager(root: Path) -> StudioJobManager:
    """Open a manager over the shared root, reconciling on construction."""
    return StudioJobManager(
        root=root, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=30.0
    )


class TestLease:
    def test_a_heartbeat_extends_a_live_lease_only(self, tmp_path: Path) -> None:
        moment = {"now": UTC_CLOCK_START}
        ledger = _ledger(tmp_path, clock=lambda: moment["now"], lease_seconds=30.0)
        _admit(ledger)
        first = ledger.record("sj_0000000000000001").lease_expires_at_utc

        moment["now"] = UTC_CLOCK_START + timedelta(seconds=10)
        ledger.heartbeat("sj_0000000000000001")
        extended = ledger.record("sj_0000000000000001").lease_expires_at_utc
        assert extended is not None and first is not None and extended > first

        ledger.transition("sj_0000000000000001", "running")
        ledger.transition("sj_0000000000000001", "completed")
        ledger.heartbeat("sj_0000000000000001")
        assert ledger.record("sj_0000000000000001").lease_expires_at_utc is None

    def test_a_positive_lease_is_required(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            StudioJobLedger(root=tmp_path, lease_seconds=0.0)


class TestSupervisorIdentity:
    def test_this_process_is_alive(self) -> None:
        assert supervisor_is_alive(supervisor_identity()) is True

    def test_a_foreign_host_cannot_be_probed(self) -> None:
        assert supervisor_is_alive("some-other-host:1:2") is None

    def test_a_malformed_identity_cannot_be_probed(self) -> None:
        assert supervisor_is_alive("nonsense") is None

    def test_a_reused_process_id_is_not_the_same_supervisor(self) -> None:
        host, pid, _token = supervisor_identity().split(":", 2)
        assert supervisor_is_alive(f"{host}:{pid}:not-the-original-token") is False


class TestReconciliationRules:
    def _admit_running(self, manager: StudioJobManager, job_id: str, **fields: object) -> None:
        with manager._ledger.transaction() as connection:
            connection.execute(
                "INSERT INTO jobs (job_id, kind, actor, workspace, request_id,"
                " idempotency_key, experiment_sha256, admission, execution_model, status,"
                " created_at_utc, artifacts, lease_owner, lease_expires_at_utc,"
                " heartbeat_at_utc, sequence)"
                " VALUES (?, 'analysis', 'alice', 'default', NULL, NULL, NULL, '{}',"
                " 'thread', 'running', '2026-09-06T00:00:00Z', '[]', ?, ?, ?, 1)",
                (
                    job_id,
                    fields["lease_owner"],
                    fields["lease_expires_at_utc"],
                    "2026-09-06T00:00:00Z",
                ),
            )

    def test_a_job_owned_by_a_live_supervisor_is_left_alone(self, tmp_path: Path) -> None:

        root = tmp_path / "jobs"
        manager = _manager(root)
        future = (datetime.now(tz=None).astimezone() + timedelta(hours=1)).isoformat()
        # A genuinely different process, still running. Using this process's own
        # identity would not test the rule: a restarting supervisor treats its
        # own identity as the incarnation that died.
        peer = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", "import time; time.sleep(300)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            peer_identity = f"{os.uname().nodename}:{peer.pid}:{_process_start_token(peer.pid)}"
            self._admit_running(
                manager,
                "sj_00000000000000aa",
                lease_owner=peer_identity,
                lease_expires_at_utc=future,
            )

            decisions = {decision.job_id: decision for decision in manager.reconcile()}
        finally:
            peer.kill()
            peer.wait(timeout=60)

        assert decisions["sj_00000000000000aa"].status == "running"
        assert "still running" in decisions["sj_00000000000000aa"].reason
        assert manager.record("sj_00000000000000aa").status == "running"

    def test_a_job_on_another_host_is_unknown_rather_than_declared_dead(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "jobs"
        manager = _manager(root)
        future = (datetime.now(tz=None).astimezone() + timedelta(hours=1)).isoformat()
        self._admit_running(
            manager,
            "sj_00000000000000bb",
            lease_owner="a-different-host:4242:1",
            lease_expires_at_utc=future,
        )

        decisions = {decision.job_id: decision for decision in manager.reconcile()}

        assert decisions["sj_00000000000000bb"].status == "unknown"
        assert "cannot be probed" in decisions["sj_00000000000000bb"].reason
        assert manager.record("sj_00000000000000bb").status == "unknown"
        # Unknown is not terminal: verification can still resolve it.
        assert (
            manager._ledger.transition(
                "sj_00000000000000bb", "interrupted", reason="verified by the operator"
            ).status
            == "interrupted"
        )

    def test_an_expired_lease_is_interrupted_even_on_this_host(self, tmp_path: Path) -> None:
        root = tmp_path / "jobs"
        manager = _manager(root)
        past = "2026-01-01T00:00:00Z"
        self._admit_running(
            manager,
            "sj_00000000000000cc",
            lease_owner="a-different-host:4242:1",
            lease_expires_at_utc=past,
        )

        decisions = {decision.job_id: decision for decision in manager.reconcile()}

        assert decisions["sj_00000000000000cc"].status == "interrupted"
        assert "expired without a heartbeat" in decisions["sj_00000000000000cc"].reason
