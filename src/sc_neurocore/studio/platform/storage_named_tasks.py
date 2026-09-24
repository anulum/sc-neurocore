# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — named storage process-task authority

"""Bind existing Studio process tasks to exact service owners and API routes.

Storage reads this static catalogue without importing or executing task code.
The policy gateway must still authorize the authenticated requester for the
selected route before any admission. A name alone grants no operation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NamedStudioTask:
    """One reviewed process task and its preserved ledger and route identities.

    ``name`` is the only selectable operation ID. ``kind`` and ``owner``
    preserve the current ledger classification. ``task_path`` is passed to a
    trusted launcher outside storage; ``routes`` constrain policy delegation.
    Constructing this value does not authorize or execute the task.
    """

    name: str
    kind: str
    owner: str
    task_path: str
    routes: tuple[str, ...]


_TASKS = (
    NamedStudioTask(
        "analysis.run",
        "analysis",
        "studio",
        "sc_neurocore.studio.api.analysis_jobs:execute_analysis_process_task",
        ("/api/analysis/jobs",),
    ),
    NamedStudioTask(
        "model.scan",
        "model_scan",
        "studio",
        "sc_neurocore.studio.api.model_scan_jobs:execute_model_scan_process_task",
        ("/api/models/scan/jobs",),
    ),
    NamedStudioTask(
        "audit.quarantine_archive",
        "evidence",
        "studio-audit-quarantine",
        "sc_neurocore.studio.api.audit_archive_jobs:execute_quarantine_archive_task",
        ("/api/studio/audit/quarantine/archive",),
    ),
    NamedStudioTask(
        "audit.quarantine_restore",
        "evidence",
        "studio-audit-quarantine-restore",
        "sc_neurocore.studio.api.audit_archive_jobs:execute_quarantine_restore_task",
        ("/api/studio/audit/quarantine/archive/restore",),
    ),
    NamedStudioTask(
        "evidence.bundle",
        "evidence",
        "studio-evidence",
        "sc_neurocore.studio.api.evidence_jobs:execute_evidence_bundle_task",
        ("/api/studio/evidence/bundle",),
    ),
    NamedStudioTask(
        "training.weight_restore",
        "training",
        "studio-training-restore",
        "sc_neurocore.studio.api.training_weight_jobs:execute_training_weight_restore_task",
        ("/api/studio/training/weight-restore",),
    ),
    NamedStudioTask(
        "compiler.compile",
        "compiler",
        "studio-compiler",
        "sc_neurocore.studio.platform.compile_process:run_compile_process_task",
        ("/api/compile",),
    ),
    NamedStudioTask(
        "compiler.model_compile",
        "compiler",
        "studio-model-compiler",
        "sc_neurocore.studio.platform.model_compile_process:run_model_compile_process_task",
        ("/api/models/compile",),
    ),
    NamedStudioTask(
        "compiler.model_cosim",
        "compiler",
        "studio-model-cosim",
        "sc_neurocore.studio.platform.model_cosim_process:run_model_cosim_process_task",
        ("/api/models/cosim",),
    ),
    NamedStudioTask(
        "compiler.pipeline",
        "compiler",
        "studio-pipeline",
        "sc_neurocore.studio.platform.pipeline_process:run_pipeline_process_task",
        ("/api/pipeline/run",),
    ),
    NamedStudioTask(
        "synthesis.run",
        "synthesis",
        "studio-synthesis",
        "sc_neurocore.studio.platform.synthesis_process:run_synthesis_process_task",
        ("/api/synth/run",),
    ),
    NamedStudioTask(
        "synthesis.multi_target",
        "synthesis",
        "studio-synthesis",
        "sc_neurocore.studio.platform.synthesis_process:run_multi_target_synthesis_process_task",
        ("/api/synth/multi-target",),
    ),
    NamedStudioTask(
        "synthesis.terminal",
        "synthesis",
        "studio-synthesis-terminal",
        "sc_neurocore.studio.platform.synthesis_process:run_synthesis_terminal_process_task",
        ("/api/synth/terminal",),
    ),
    NamedStudioTask(
        "synthesis.pnr",
        "synthesis",
        "studio-pnr",
        "sc_neurocore.studio.platform.synthesis_process:run_pnr_process_task",
        ("/api/synth/pnr",),
    ),
    NamedStudioTask(
        "training.start",
        "training",
        "studio-training",
        "sc_neurocore.studio.platform.training_process:run_training_process_task",
        ("/api/training/start",),
    ),
    NamedStudioTask(
        "training.attach",
        "training",
        "studio-training-attach",
        "sc_neurocore.studio.platform.training_process:run_training_attach_process_task",
        (
            "/api/studio/training/weight-restore/attach",
            "/api/studio/training/weight-restore/attach/live",
        ),
    ),
)

_BY_NAME = {task.name: task for task in _TASKS}


def resolve_named_studio_task(name: str, *, authorized_route: str) -> NamedStudioTask:
    """Return one exact reviewed task for an already authorized POST route.

    Unknown names and route/task mismatches refuse before any capacity or
    filesystem mutation. The returned ``task_path`` is metadata for a trusted
    launcher, never a module imported by the storage authority.

    Parameters
    ----------
    name : str
        Exact operation identifier supplied by the trusted API adapter.
    authorized_route : str
        Route authenticated and authorized by the service policy gateway.

    Returns
    -------
    NamedStudioTask
        Preserved kind, service owner and launcher task for that route.

    Raises
    ------
    ValueError
        Name, route or their exact pairing is not in the reviewed catalogue.
    """
    if not isinstance(name, str) or not isinstance(authorized_route, str):
        raise ValueError("invalid named Studio task selection")
    task = _BY_NAME.get(name)
    if task is None or authorized_route not in task.routes:
        raise ValueError("named Studio task is not available on this route")
    return task


def named_studio_task_for_path(task_path: str, *, authorized_route: str) -> NamedStudioTask:
    """Return the reviewed task that runs ``task_path`` on an authorised route.

    The isolated API facade keeps the embedded ``submit_process_task``
    signature, whose callers name the task by import path; only a path that
    the catalogue names for this exact route becomes a named submission.

    Raises
    ------
    ValueError
        No reviewed task runs that path on this route.
    """
    for task in _TASKS:
        if task.task_path == task_path and authorized_route in task.routes:
            return task
    raise ValueError("named Studio task is not available on this route")
