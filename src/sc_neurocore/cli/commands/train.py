# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Command-line Studio training run

"""Run one Studio training job from the command line and report its verdict.

The request is the body ``POST /api/training/start`` accepts and runs through
the same contract, job manager and worker; the job ledger, sandboxes and
sealed artefacts are kept under ``--job-root``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Exit status of a run that completed but missed its preregistered criterion.
EXIT_CRITERION_MISSED = 3


def add_train_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register ``train``.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "train",
        help="Run one Studio training job and report its preregistered verdict",
        description=(
            "Run a training request through the Studio contract and job manager, wait for "
            "it to finish and print its status, metrics and verdict as JSON. Exit status: 0 "
            "completed (criterion met or none declared), 1 failed or stopped, 2 request "
            f"refused, {EXIT_CRITERION_MISSED} completed but the preregistered criterion was missed."
        ),
    )
    parser.add_argument(
        "config", help="Training request JSON, as POST /api/training/start takes it"
    )
    parser.add_argument(
        "--job-root",
        required=True,
        help="Directory for the job ledger, sandboxes and sealed artefacts",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Seconds the job may run before it is timed out (default: the Studio default)",
    )
    parser.set_defaults(handler=run_train)


def run_train(args: argparse.Namespace) -> int:
    """Run one training request to a terminal state and print its outcome.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``config``, ``job_root`` and ``timeout``.

    Returns
    -------
    int
        0 completed with its criterion met or none declared, 1 failed or
        stopped, 2 request refused, 3 criterion missed.
    """
    from sc_neurocore.studio.api.runtime import DEFAULT_STUDIO_JOB_KINDS
    from sc_neurocore.studio.platform.jobs_manager import StudioJobManager
    from sc_neurocore.studio.platform.settings import (
        DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
        DEFAULT_STUDIO_JOB_TIMEOUT_SECONDS,
    )
    from sc_neurocore.studio.training import get_training_status, start_training
    from sc_neurocore.studio.training_contract import TrainingConfigError

    try:
        request = json.loads(Path(args.config).read_text(encoding="utf-8"))
        manager = StudioJobManager(
            root=Path(args.job_root),
            allowed_kinds=DEFAULT_STUDIO_JOB_KINDS,
            default_timeout_seconds=(
                DEFAULT_STUDIO_JOB_TIMEOUT_SECONDS if args.timeout is None else args.timeout
            ),
            max_artifact_bytes=DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
            configured=True,
        )
        started = start_training(request, manager)
    except TrainingConfigError as refusal:
        print(json.dumps(refusal.to_public_detail(), sort_keys=True), file=sys.stderr)
        return 2
    except (OSError, ValueError) as refusal:
        print(
            json.dumps({"error": "training_request_unreadable", "reason": str(refusal)}),
            file=sys.stderr,
        )
        return 2
    job_id = str(started["job_id"])
    manager.wait(job_id)
    status = get_training_status(job_id, manager)
    outcome = {
        key: status.get(key)
        for key in (
            "job_id",
            "status",
            "error",
            "final_metrics",
            "preregistration_verdict",
            "weight_checkpoint",
        )
    }
    outcome["job_root"] = str(Path(args.job_root).resolve())
    print(json.dumps(outcome, sort_keys=True))
    if status.get("status") != "completed":
        return 1
    verdict = status.get("preregistration_verdict")
    if isinstance(verdict, dict) and verdict.get("passed") is not True:
        return EXIT_CRITERION_MISSED
    return 0


__all__ = ["EXIT_CRITERION_MISSED", "add_train_command", "run_train"]
