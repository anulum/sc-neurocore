# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit storage boundary configuration

"""Validate isolated boundary intent without changing files or enabling storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_Uid = Annotated[int, Field(gt=0, lt=0xFFFFFFFF)]


class StorageBoundaryConfiguration(BaseModel):
    """Immutable role, path and resource intent, not proof of OS isolation.

    All fields are required. Three non-root OS identities must differ. Storage,
    spool and socket-parent trees must be canonical absolute disjoint paths.
    Limits carry explicit frame, metadata, seed, artefact, transfer and connection
    budgets.
    Actual ownership, ACLs, launcher and endpoint lifecycle are checked separately
    before a future isolated runtime can create its ledger or listener.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    storage_uid: _Uid
    api_uid: _Uid
    worker_uid: _Uid
    authority_root: Path
    spool_root: Path
    socket_path: Path
    workspace: Annotated[str, Field(min_length=1)]
    frame_max_bytes: Annotated[int, Field(gt=0, le=0xFFFFFFFF)]
    max_metadata_bytes: Annotated[int, Field(gt=0, le=0xFFFFFFFF)]
    max_seed_bytes: Annotated[int, Field(ge=0, le=0xFFFFFFFF)]
    max_seed_entries: Annotated[int, Field(ge=0, le=0xFFFFFFFF)]
    max_manifest_bytes: Annotated[int, Field(gt=0, le=0xFFFFFFFF)]
    max_artifact_bytes: Annotated[int, Field(ge=0, le=0xFFFFFFFF)]
    max_artifact_entries: Annotated[int, Field(ge=0, le=0xFFFFFFFF)]
    transfer_timeout_seconds: Annotated[float, Field(gt=0)]
    max_connections: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        """Refuse collapsed identities and overlapping or aliased namespaces.

        Returns
        -------
        StorageBoundaryConfiguration
            This immutable boundary when shape and disjointness are valid.

        Raises
        ------
        ValueError
            Role identities coincide, workspace is blank, or paths are root,
            relative, noncanonical, inaccessible to resolution or overlapping.

        Notes
        -----
        Path resolution observes existing links but never creates directories.
        It does not prevent later mutation or replace startup permission checks.
        """
        if len({self.storage_uid, self.api_uid, self.worker_uid}) != 3:
            raise ValueError("storage, API and worker identities must differ")
        if not self.workspace.strip():
            raise ValueError("storage workspace must not be blank")
        if self.max_metadata_bytes > self.frame_max_bytes:
            raise ValueError("storage metadata limit exceeds frame limit")
        if self.max_manifest_bytes > self.max_metadata_bytes:
            raise ValueError("storage manifest limit exceeds metadata limit")
        for path in (self.authority_root, self.spool_root, self.socket_path):
            try:
                valid = _canonical(path)
            except (OSError, RuntimeError) as exc:
                raise ValueError("storage boundary path cannot be resolved") from exc
            if not valid:
                raise ValueError("storage boundary paths must be canonical absolute non-root paths")
        roots = (self.authority_root, self.spool_root, self.socket_path.parent)
        for index, root in enumerate(roots):
            for other in roots[index + 1 :]:
                if root.is_relative_to(other) or other.is_relative_to(root):
                    raise ValueError("storage, spool and socket parent must not overlap")
        return self


def _canonical(path: Path) -> bool:
    """Return whether ``path`` is absolute, not a root, and passes through no link.

    ``Path.resolve`` alone does not decide this: from Python 3.13 it returns a
    path through a symlink loop unchanged instead of raising, so every existing
    component is also checked for being a link.
    """
    if not path.is_absolute() or path == Path(path.anchor) or path.resolve() != path:
        return False
    return not any(component.is_symlink() for component in (path, *path.parents))


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate storage boundary field")
        result[key] = value
    return result


def parse_storage_boundary(value: str | None) -> StorageBoundaryConfiguration | None:
    """Decode explicit operator JSON without defaults or persistent side effects.

    Parameters
    ----------
    value : str or None
        Boundary JSON from trusted configuration; absence preserves no boundary.

    Returns
    -------
    StorageBoundaryConfiguration or None
        Validated intent, never evidence of a launched isolated service.

    Raises
    ------
    ValueError
        JSON, duplicate fields, types, identity, paths or limits are invalid.
    """
    if value is None:
        return None
    try:
        json.loads(value, object_pairs_hook=_unique_fields)
        return StorageBoundaryConfiguration.model_validate_json(value, strict=True)
    except RecursionError as exc:
        raise ValueError("storage boundary JSON nesting is invalid") from exc
