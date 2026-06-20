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
import threading
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
    result_path.write_text(
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


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests.
    raise SystemExit(main())
