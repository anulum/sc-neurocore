# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio process-worker test tasks

"""Import-stable tasks used by Studio process-worker tests."""

from __future__ import annotations

import time
from collections.abc import Mapping

from sc_neurocore.studio.platform.jobs import StudioJobContext

NON_CALLABLE_TASK = 1


def process_echo_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Write one manifest-declared artifact and echo the JSON payload.

    Parameters
    ----------
    context:
        Studio job context that owns the process worker's confined work
        directory.
    payload:
        JSON object loaded by the process-worker entrypoint.

    Returns
    -------
    dict[str, object]
        Path-free payload echo and worker job identifier.
    """

    context.write_artifact("reports/process-result.txt", "process ok")
    return {"payload": dict(payload), "worker_job_id": context.job_id}


def process_sleep_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Sleep for a requested duration to exercise timeout and cancel paths.

    Parameters
    ----------
    context:
        Studio job context supplied by the process worker. The task does not
        need to write artifacts.
    payload:
        JSON object containing an optional numeric ``seconds`` value.

    Returns
    -------
    dict[str, object]
        Path-free completion marker.
    """

    del context
    seconds = payload.get("seconds")
    time.sleep(float(seconds) if isinstance(seconds, int | float) else 1.0)
    return {"slept": True}


def process_failure_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Raise a stable exception for process-worker failure tests.

    Parameters
    ----------
    context:
        Studio job context supplied by the process worker.
    payload:
        JSON object supplied by the process worker.

    Returns
    -------
    dict[str, object]
        This task never returns.

    Raises
    ------
    ValueError
        Always raised with a deterministic message.
    """

    del context, payload
    raise ValueError("hidden local failure detail")


__all__ = [
    "NON_CALLABLE_TASK",
    "process_echo_task",
    "process_failure_task",
    "process_sleep_task",
]
