# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage admission content digest

"""Derive durable replay identity from validated content and received seeds."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_seed_files import ReceivedStorageSeedFile
from sc_neurocore.studio.platform.storage_named_tasks import (
    NamedStudioTask,
    resolve_named_studio_task,
)


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON names at every parsed object depth."""
    fields: dict[str, object] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate storage admission field")
        fields[key] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions before canonical serialization."""
    raise ValueError("nonfinite storage admission JSON constant")


def _valid_timeout(value: object, *, allow_zero: bool) -> bool:
    """Validate finite timing without overflowing on arbitrary JSON integers."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
    except OverflowError:
        return False


def derive_storage_admission_replay(
    *,
    requester: Principal,
    mutation_id: str,
    workspace: str,
    task: NamedStudioTask,
    authorized_route: str,
    payload_json: bytes,
    seed_inputs: Mapping[str, bytes | ReceivedStorageSeedFile],
    execution_timeout_seconds: float,
    queue_wait_seconds: float | None,
    admission: Mapping[str, object] | None,
    training_config: Mapping[str, object] | None,
    experiment_sha256: str | None,
    max_metadata_bytes: int,
    max_seed_bytes: int,
    max_seed_entries: int,
) -> StorageAdmissionReplay:
    """Hash one exact named admission without trusting a caller-supplied digest.

    ``task`` must match the service's reviewed named-operation map and its
    authorized route. ``requester`` must come from
    its policy-allowed trusted API peer, and ``workspace`` from configuration.
    Seeds must already have passed bounded ingress; this function hashes
    their actual content and refuses aggregate bytes beyond ``max_seed_bytes``.
    The caller's HTTP trace, mutation ID and process generation are excluded
    from content so a lost reply can replay after an API restart. This function
    prepares the replay key; it does not authorize, admit or launch a worker.

    Parameters
    ----------
    requester : Principal
        Authenticated principal delegated by the trusted API peer.
    mutation_id : str
        Durable retry key, distinct from the HTTP trace identifier.
    workspace : str
        Server-configured workspace, never a browser-selected scope.
    task : NamedStudioTask
        Reviewed task selected for ``authorized_route``.
    authorized_route : str
        Route whose existing policy the service has allowed.
    payload_json : bytes
        Bounded UTF-8 JSON object from the admitted request.
    seed_inputs : mapping of str to bytes or ReceivedStorageSeedFile
        Actual received seed bytes or private staged files. Staged files are
        rehashed in bounded chunks; their declared digests are not trusted.
    execution_timeout_seconds, queue_wait_seconds : float or None
        Worker deadline and distinct admission queue deadline.
    admission, training_config : mapping or None
        Existing job admission and validated training snapshot controls.
    experiment_sha256 : str or None
        Effective experiment digest, when one exists.
    max_metadata_bytes, max_seed_bytes, max_seed_entries : int
        Explicit service limits for canonical content and received seeds.

    Returns
    -------
    StorageAdmissionReplay
        Validated requester, mutation key and service-derived SHA-256 digest.

    Raises
    ------
    ValueError
        Invalid identity, JSON, nonfinite value, size or seed content.
    """
    if (
        not isinstance(requester, Principal)
        or type(max_metadata_bytes) is not int
        or max_metadata_bytes <= 0
        or type(max_seed_bytes) is not int
        or max_seed_bytes < 0
        or type(max_seed_entries) is not int
        or max_seed_entries < 0
        or not _valid_timeout(execution_timeout_seconds, allow_zero=False)
        or (
            queue_wait_seconds is not None
            and not _valid_timeout(queue_wait_seconds, allow_zero=True)
        )
    ):
        raise ValueError("invalid storage admission content limits")
    if not isinstance(workspace, str) or not workspace or not workspace.isprintable():
        raise ValueError("invalid storage admission workspace")
    if not isinstance(task, NamedStudioTask):
        raise ValueError("invalid named Studio task")
    reviewed_task = resolve_named_studio_task(task.name, authorized_route=authorized_route)
    if task != reviewed_task:
        raise ValueError("named Studio task differs from reviewed operation")
    if not isinstance(payload_json, bytes) or not 0 < len(payload_json) <= max_metadata_bytes:
        raise ValueError("storage admission payload exceeds metadata limit")
    try:
        payload = json.loads(
            payload_json.decode("utf-8"),
            object_pairs_hook=_unique_fields,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage admission payload JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("storage admission payload must be an object")
    if not isinstance(seed_inputs, Mapping):
        raise ValueError("storage admission seeds must be a mapping")
    if len(seed_inputs) > max_seed_entries:
        raise ValueError("storage admission seeds exceed entry limit")
    if any(not isinstance(name, str) or not name or not name.isprintable() for name in seed_inputs):
        raise ValueError("invalid storage admission seed name")
    seeds: list[dict[str, object]] = []
    total = 0
    for name, data in sorted(seed_inputs.items()):
        if isinstance(data, bytes):
            size = len(data)
            seed_sha256 = hashlib.sha256(data).hexdigest()
        elif isinstance(data, ReceivedStorageSeedFile):
            if data.name != name or type(data.size) is not int or data.size < 0:
                raise ValueError("storage admission staged seed identity is invalid")
            if total + data.size > max_seed_bytes:
                raise ValueError("storage admission seeds exceed aggregate limit")
            digest = hashlib.sha256()
            size = 0
            data.stream.seek(0)
            while chunk := data.stream.read(64 * 1024):
                if not isinstance(chunk, bytes):
                    raise ValueError("storage admission staged seed is not binary")
                size += len(chunk)
                if total + size > max_seed_bytes:
                    raise ValueError("storage admission seeds exceed aggregate limit")
                digest.update(chunk)
            data.stream.seek(0)
            seed_sha256 = digest.hexdigest()
            if size != data.size or seed_sha256 != data.sha256:
                raise ValueError("storage admission staged seed changed after receipt")
        else:
            raise ValueError("storage admission seed must contain received bytes")
        total += size
        if total > max_seed_bytes:
            raise ValueError("storage admission seeds exceed aggregate limit")
        seeds.append({"name": name, "size": size, "sha256": seed_sha256})
    roles = requester.roles
    if not requester.principal_id or any(
        not isinstance(role, str) or not role or not role.isprintable() for role in roles
    ):
        raise ValueError("invalid storage admission requester")
    content = {
        "schema_version": "studio.storage.admission-content.v1",
        "requester": {"principal_id": requester.principal_id, "roles": sorted(roles)},
        "workspace": workspace,
        "kind": task.kind,
        "owner": task.owner,
        "named_task": task.name,
        "task_path": task.task_path,
        "authorized_route": authorized_route,
        "payload": payload,
        "seeds": seeds,
        "execution_timeout_seconds": execution_timeout_seconds,
        "queue_wait_seconds": queue_wait_seconds,
        "admission": admission,
        "training_config": training_config,
        "experiment_sha256": experiment_sha256,
    }
    try:
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
        json.loads(canonical, object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
        encoded = canonical.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid storage admission content JSON") from exc
    if len(encoded) > max_metadata_bytes:
        raise ValueError("storage admission content exceeds metadata limit")
    replay = StorageAdmissionReplay(
        requester=requester.principal_id,
        mutation_id=mutation_id,
        payload_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    replay.validate()
    return replay
