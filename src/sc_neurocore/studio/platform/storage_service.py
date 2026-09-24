# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage authority service entry

"""Run the Studio storage authority as an operator-installed service.

The service opens the ledger in its configured authority root, completes any
purge a previous run left committed, and then serves one verified request at
a time through the listener, whose startup checks ownership and modes and
never repairs them. While idle it reconciles on a fixed interval: jobs whose
delegated API generation is provably gone are resolved by the ledger's own
rules, and capacity is released only where the stored identity proves the
work stopped. A refused or malformed connection is closed and the service
continues; SIGTERM or SIGINT stops it after the current request.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import signal
import time
from types import FrameType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_purge_recovery import recover_purges
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.policy_audit import JsonlAuditSink
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_listener import StorageRecordListener
from sc_neurocore.studio.platform.storage_named_admit import named_process_admission
from sc_neurocore.studio.platform.storage_purge import AuthorityCustody


class StorageServiceConfiguration(BaseModel):
    """The boundary plus the service-owned admission limits, audit and cadence."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    boundary: StorageBoundaryConfiguration
    max_concurrent: Annotated[int, Field(ge=1, le=4096)]
    max_queued: Annotated[int, Field(ge=0, le=65536)]
    audit_log_path: Path
    reconcile_seconds: Annotated[float, Field(gt=0, le=3600)]


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage service field")
        fields[name] = value
    return fields


def load_service_configuration(path: Path) -> StorageServiceConfiguration:
    """Read and strictly validate the service configuration file.

    Raises
    ------
    ValueError
        JSON, duplicate fields, types, values or the boundary are invalid.
    OSError
        The file cannot be read.
    """
    text = path.read_text(encoding="utf-8")
    try:
        json.loads(text, object_pairs_hook=_unique_fields)
    except RecursionError as exc:
        raise ValueError("storage service JSON nesting is invalid") from exc
    return StorageServiceConfiguration.model_validate_json(text, strict=True)


class StorageService:
    """The authority's ledger, admission, gateway and listener for one run."""

    def __init__(self, configuration: StorageServiceConfiguration) -> None:
        """Open the ledger and collaborators; nothing is bound yet."""
        boundary = configuration.boundary
        self._configuration = configuration
        self.ledger = StudioJobLedger(
            root=boundary.authority_root, supervisor=supervisor_identity()
        )
        self.admission = SharedJobAdmission(
            self.ledger,
            max_concurrent=configuration.max_concurrent,
            max_queued=configuration.max_queued,
        )
        self.listener = StorageRecordListener(
            boundary,
            ledger=self.ledger,
            gateway=PolicyGateway(JsonlAuditSink(configuration.audit_log_path)),
            admit_named=named_process_admission(self.admission, workspace=boundary.workspace),
            admission=self.admission,
        )
        self._reconciled_at = float("-inf")

    def reconcile(self) -> None:
        """Finish committed purges, resolve abandoned jobs, release proven capacity."""
        recover_purges(AuthorityCustody(self.ledger))
        self.ledger.reconcile()
        self.admission.reconcile()
        self._reconciled_at = time.monotonic()

    def serve_once(self) -> None:
        """Serve one request, or reconcile when idle and the interval elapsed."""
        try:
            self.listener.serve_once()
        except TimeoutError:
            if time.monotonic() - self._reconciled_at >= self._configuration.reconcile_seconds:
                self.reconcile()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the storage authority until SIGTERM or SIGINT.

    Parameters
    ----------
    argv : Sequence[str] or None
        ``--configuration PATH``; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` after an orderly stop. Startup refusal raises instead.
    """
    parser = argparse.ArgumentParser(prog="studio-storage-service")
    parser.add_argument("--configuration", required=True)
    args = parser.parse_args(argv)
    service = StorageService(load_service_configuration(Path(args.configuration)))
    stopping = False

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    service.reconcile()
    with service.listener:
        print("ready", flush=True)
        while not stopping:
            try:
                service.serve_once()
            except (PermissionError, ValueError, EOFError, OSError):
                continue
    service.ledger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
