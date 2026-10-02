# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Importable real job failure workloads

"""Exercise public job artifact operations from actual supervised interpreters."""

import json
import os
import signal
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import FrameType

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger


def write_oversized_artifact(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Reach the production artifact byte limit with a real file-write request."""
    context.write_artifact("report.txt", b"12345")
    return {}


def write_unnameable_artifact(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Make the filesystem itself refuse an overlong file component."""
    context.write_artifact("reports/" + "x" * 300, b"data")
    return {}


def invalid_worker_output(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Exercise nonfinite, absent or malformed output from an actual worker.

    Parameters
    ----------
    context : StudioJobContext
        Actual worker artifact context.
    payload : Mapping[str, object]
        Selects the invalid output representation.

    Returns
    -------
    dict[str, object]
        A nonfinite result, unless the worker exits with absent or invalid bytes.
    """
    if payload["form"] == "nonfinite":
        context.check_cancelled()
        return {"observation": float("nan")}
    if payload["form"] != "missing":
        raw = b"[]" if payload["form"] == "nonobject" else b"\xff"
        context.write_artifact(".studio_process_result.json", raw)
    os._exit(0)


def restore_record_on_cancellation(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Repair an operator-removed row after actual cancellation reaches the worker.

    Parameters
    ----------
    context : StudioJobContext
        Real supervised worker context.
    payload : Mapping[str, object]
        Owned ledger root and thread or process execution mode.

    Returns
    -------
    dict[str, object]
        Confirmation that the captured row was restored through real SQLite.
    """
    root = payload["root"]
    if not isinstance(root, str):
        raise TypeError("The recovery workload needs its owned ledger root.")
    stopped = threading.Event()

    def stop(number: int, frame: FrameType | None) -> None:
        stopped.set()

    if payload["mode"] == "process":
        signal.signal(signal.SIGTERM, stop)
    context.write_artifact("ready.txt", "worker ready")
    deadline = time.monotonic() + 30.0
    while not context.cancelled and not stopped.is_set():
        if time.monotonic() >= deadline:
            raise TimeoutError("No actual cancellation reached the recovery workload.")
        time.sleep(0.005)
    captured = json.loads((Path(root) / context.job_id / ".restore_row.json").read_text())
    if not isinstance(captured, dict):
        raise TypeError("The captured row must be an object.")
    columns = ", ".join(captured)
    placeholders = ", ".join("?" for _ in captured)
    ledger = StudioJobLedger(root=Path(root))
    try:
        with ledger.transaction() as connection:
            connection.execute(
                f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", tuple(captured.values())
            )
    finally:
        ledger.close()
    context.write_artifact("restored.txt", "original row restored")
    return {"restored": True}
