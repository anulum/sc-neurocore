# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Purging complete event training records

"""Delete a stopped event run through the public manager and real authority."""

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from tests.studio_storage_generation_support import Authority
from tests.test_studio_storage_event_stop import exercise_event_stop
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_event_training import launcher as launcher
from tests.test_studio_storage_event_training import event_config as event_config
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.test_studio_storage_isolated_jobs import manager as manager
from tests.test_studio_storage_isolated_jobs import request_on


def test_purging_a_stopped_event_run_returns_its_complete_pre_purge_record(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    authority: Authority,
    event_config: dict[str, object],
) -> None:
    """A confirmed stopped worker loses its archive, while its reply retains input."""
    finished = exercise_event_stop(manager, ledger, authority, event_config, "cancel")
    archive = ledger.path.parent / finished.job_id
    assert archive.is_dir()
    with request_on("/api/studio/audit/quarantine/archive/purge"):
        assert manager.purge_terminal_record(finished.job_id) == finished
    with request_on("/api/studio/jobs", "GET"):
        with pytest.raises(KeyError):
            manager.record(finished.job_id)
        assert manager.list_records() == ()
    assert ledger.list_records() == ()
    assert not archive.exists()
    assert not (archive.parent / f".purge-{finished.job_id}").exists()
    assert ledger.connection().execute("SELECT COUNT(*) FROM job_purges").fetchone()[0] == 0
