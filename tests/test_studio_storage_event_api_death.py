# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event worker recovery after API process death

"""Keep the authority alive while a real API generation dies during training."""

import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import threading
import time

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_worker_recovery import worker_group_stopped
from sc_neurocore.studio.platform.storage_dispatch import serve_operation
from sc_neurocore.studio.platform.storage_peer import read_verified_frame
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from tests.studio_storage_generation_runs import api_runtime
from tests.studio_storage_generation_support import Authority, FRAME
from tests.studio_storage_launcher_support import Launcher
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_event_training import event_config as event_config
from tests.test_studio_storage_event_training import launcher as launcher
from tests.studio_storage_generation_runs import ledger as ledger
from tests.test_studio_storage_isolated_jobs import _configuration, request_on


def run_api_generation(base: Path) -> None:
    """Submit actual event training from an independent API process."""
    from sc_neurocore.studio.platform.storage_generation_exchanges import GenerationRuntime
    from sc_neurocore.studio.platform.storage_live_spool import LiveSpools
    from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK

    def connect() -> socket.socket:
        """Keep kernel peer identity intact on the real authority endpoint."""
        channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        channel.connect(str(base / "sock" / "authority.sock"))
        return channel

    runtime = GenerationRuntime(
        workspace="default",
        spool_root=base / "spool",
        storage_uid=os.getuid(),
        frame_max_bytes=FRAME,
        transfer_timeout_seconds=10,
        launcher_socket=base / "sock" / "launcher.sock",
        launcher_uid=os.getuid(),
        worker_uid=os.getuid(),
        worker_gid=os.getgid(),
        max_artifact_bytes=1 << 20,
        artifact_total_bytes=1 << 20,
        artifact_entries=64,
        grant_timeout_seconds=60,
        heartbeat_seconds=0.2,
        poll_seconds=0.05,
        attempts=3,
        connect=connect,
        live=LiveSpools(retain=4, max_seed_bytes=1 << 20),
    )
    manager = IsolatedJobManager(
        runtime,
        _configuration(base),
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=120,
    )
    config = json.loads((base / "training.json").read_text())
    with request_on("/api/training/start"):
        record = manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="event-api-death",
            task_path=TRAINING_PROCESS_TASK,
            payload=config,
            training_config=config,
        )
    print(record.job_id, flush=True)
    time.sleep(180)


def test_api_death_interrupts_real_event_worker_without_losing_its_contract(
    base: Path,
    launcher: Launcher,
    ledger: StudioJobLedger,
    event_config: dict[str, object],
) -> None:
    """Independent authority retains full input and releases only proven stopped work."""
    config = {**event_config, "epochs": 10000}
    (base / "training.json").write_text(json.dumps(config))
    authority = Authority(ledger)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(base / "sock" / "authority.sock"))
    listener.listen(16)
    listener.settimeout(0.2)
    stopping = threading.Event()
    failures: list[BaseException] = []

    def serve() -> None:
        """Serve actual handler frames with the original kernel peer connection."""
        try:
            while not stopping.is_set():
                try:
                    channel, _ = listener.accept()
                except TimeoutError:
                    continue
                try:
                    with channel:
                        deadline = time.monotonic() + 10
                        frame = read_verified_frame(
                            channel, expected_uid=os.getuid(), max_bytes=FRAME, deadline=deadline
                        )
                        serve_operation(channel, frame, authority.services, deadline=deadline)
                except (PermissionError, ValueError, EOFError, OSError) as exc:
                    failures.append(exc)
        finally:
            ledger.close()

    server = threading.Thread(target=serve)
    server.start()
    api = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; from tests.test_studio_storage_event_api_death import run_api_generation; run_api_generation(Path(sys.argv[1]))",
            str(base),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")
            + os.pathsep
            + str(Path(__file__).resolve().parents[1]),
        },
    )
    try:
        assert api.stdout is not None
        assert select.select([api.stdout], [], [], 60)[0]
        job_id = api.stdout.readline().decode().strip()
        if not job_id.startswith("sj_"):
            api.wait(timeout=10)
            assert api.stderr is not None
            raise AssertionError(api.stderr.read().decode())
        deadline = time.monotonic() + 60
        events: list[Path] = []
        while time.monotonic() < deadline:
            events = list((base / "spool" / job_id).glob("*/*/training/events.jsonl"))
            if events and events[0].stat().st_size:
                break
            assert api.poll() is None
            time.sleep(0.05)
        assert events and events[0].stat().st_size
        assert ledger.record(job_id).status == "running"
        assert failures == []
        row = (
            ledger.connection()
            .execute(
                "SELECT worker_identity,boot_id,group_id FROM job_workers WHERE job_id=?", (job_id,)
            )
            .fetchone()
        )
        assert row is not None
        assert not worker_group_stopped(str(row[0]), str(row[1]), int(row[2]))
        api.kill()
        api.wait(timeout=10)
        deadline = time.monotonic() + 20
        while not worker_group_stopped(str(row[0]), str(row[1]), int(row[2])):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        decisions = ledger.reconcile()
        assert any(item.job_id == job_id and item.status == "interrupted" for item in decisions)
        assert authority.services.admission is not None
        assert job_id in authority.services.admission.reconcile()
        recovered = IsolatedJobManager(
            api_runtime(base, authority),
            _configuration(base),
            allowed_kinds=frozenset({"training"}),
            default_timeout_seconds=120,
        )
        with request_on("/api/studio/jobs", "GET"):
            record = recovered.record(job_id)
            assert recovered.list_records() == (record,)
        assert record.status == "interrupted"
        assert record.training_config == config
        assert record.result is None
        assert record.artifacts == ()
        with request_on("/api/studio/jobs/status", "GET"):
            assert recovered.unreaped_workers == ()
        assert launcher.process.poll() is None
    finally:
        if api.poll() is None:
            api.kill()
        api.wait(timeout=10)
        for stream in (api.stdout, api.stderr):
            if stream is not None:
                stream.close()
        stopping.set()
        server.join(timeout=15)
        assert not server.is_alive()
        listener.close()
        authority.join()
