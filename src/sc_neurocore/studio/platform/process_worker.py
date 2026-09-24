# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio isolated process worker

"""Subprocess entrypoint for isolated SC-NeuroCore Studio jobs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import threading
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

from sc_neurocore.studio.platform.jobs import StudioJobContext

_ProcessTask = Callable[[StudioJobContext, Mapping[str, object]], dict[str, object]]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one importable Studio process task and persist a JSON result.

    Parameters
    ----------
    argv:
        Optional command-line argument sequence. ``None`` reads process
        arguments from ``sys.argv`` through ``argparse``.

    Returns
    -------
    int
        ``0`` when the imported task completed and wrote a result; ``1`` when
        the task failed and the result file contains the public error string.
    """

    args = _parse_args(argv)
    result_path = Path(args.result)
    try:
        if (args.grant_socket is None) != (args.grant_server_uid is None):
            raise ValueError("A socket grant needs both its endpoint and server identity.")
        if args.grant_socket is not None and args.supervisor is None:
            raise ValueError("A socket grant requires a named supervisor.")
        if args.supervisor is not None:
            from sc_neurocore.studio.platform.jobs_worker_guard import arm_worker_guard

            if args.grant_socket is None:
                from sc_neurocore.studio.platform.jobs_worker_registration import (
                    await_worker_registration,
                )

                with sys.stdin.buffer:
                    await_worker_registration(sys.stdin.buffer.fileno())
            else:
                from sc_neurocore.studio.platform.storage_worker_grant import (
                    receive_socket_grant,
                )

                receive_socket_grant(
                    Path(args.grant_socket), expected_server_uid=args.grant_server_uid
                )
            guard = arm_worker_guard(args.supervisor)
            if guard.poll() is not None:
                raise RuntimeError("Worker lifetime guard exited before task startup.")
        payload = _load_payload(Path(args.payload))
        task = _load_task(args.task)
        context = StudioJobContext(
            job_id=Path(args.work_dir).name,
            work_dir=Path(args.work_dir),
            cancel_event=threading.Event(),
            max_artifact_bytes=args.max_artifact_bytes,
        )
        result = task(context, payload)
        _write_result(
            result_path,
            status="completed",
            result=result,
            error=None,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001 - persisted as job failure state.
        _write_failure_result(result_path, type(exc).__name__)
        return 1
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="studio-process-worker")
    parser.add_argument("--task", required=True)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    parser.add_argument("--supervisor", help="Expected supervisor for ledger-managed execution")
    parser.add_argument(
        "--grant-socket",
        help="API grant endpoint for a launcher-started worker, replacing the stdin grant",
    )
    parser.add_argument(
        "--grant-server-uid",
        type=int,
        help="Configured API identity that must own the grant endpoint",
    )
    return parser.parse_args(argv)


def _load_payload(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Studio process payload must be a JSON object.")
    return cast(dict[str, object], payload)


def _load_task(task_path: str) -> _ProcessTask:
    module_path, _, function_name = task_path.partition(":")
    module = importlib.import_module(module_path)
    task = getattr(module, function_name)
    if not callable(task):
        raise TypeError("Studio process task import did not resolve to a callable.")
    return cast(_ProcessTask, task)


def _write_failure_result(result_path: Path, error: str) -> None:
    _write_result(
        result_path,
        status="failed",
        result={},
        error=error,
        context=None,
    )


def _write_result(
    result_path: Path,
    *,
    status: str,
    result: dict[str, object],
    error: str | None,
    context: StudioJobContext | None,
) -> None:
    """Publish the result whole: a worker killed while writing leaves no result.

    The supervisor reads the result after stopping the worker; a truncated file
    would turn its own verdict into an unreadable-output failure.
    """
    partial = result_path.with_name(f"{result_path.name}.partial")
    partial.write_text(
        json.dumps(
            {
                "artifacts": []
                if context is None
                else [artifact.to_public_dict() for artifact in context.artifacts],
                "error": error,
                "result": result,
                "status": status,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(partial, result_path)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests.
    raise SystemExit(main())
