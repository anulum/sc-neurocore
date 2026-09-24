# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launcher client settings of the isolated API

"""Validated operator settings for reaching the launcher from the isolated API.

Kept free of runtime imports so the Studio settings can parse them without
loading the storage clients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

_Positive = Annotated[float, Field(gt=0, le=3600)]


class LauncherClientSettings(BaseModel):
    """How the API reaches the launcher and supervises launched generations."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    socket_path: Path
    # The privileged launcher that switches workers to the compute identity
    # runs as root, so UID 0 is a valid launcher identity.
    launcher_uid: Annotated[int, Field(ge=0, lt=0xFFFFFFFF)]
    worker_gid: Annotated[int, Field(gt=0, lt=0xFFFFFFFF)]
    grant_timeout_seconds: _Positive
    heartbeat_seconds: _Positive
    poll_seconds: _Positive
    attempts: Annotated[int, Field(ge=1, le=16)]
    live_retain: Annotated[int, Field(ge=0, le=4096)]
    # The only launcher backend is direct spawning, which proves neither the
    # termination of every forked or detached descendant nor per-job
    # resource accounting. The isolated API refuses to start unless the
    # operator explicitly accepts those limits, as a qualification run does.
    accept_direct_backend_limits: bool

    @field_validator("socket_path")
    @classmethod
    def canonical_socket(cls, value: Path) -> Path:
        """Accept only an absolute canonical endpoint path."""
        if not value.is_absolute() or value.resolve() != value:
            raise ValueError("launcher socket path must be absolute and canonical")
        return value


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate launcher client field")
        fields[name] = value
    return fields


def parse_launcher_client(value: str | None) -> LauncherClientSettings | None:
    """Decode operator JSON for the launcher client; absence means none.

    Raises
    ------
    ValueError
        JSON, duplicate fields, types or values are invalid.
    """
    if value is None:
        return None
    try:
        json.loads(value, object_pairs_hook=_unique_fields)
    except RecursionError as exc:
        raise ValueError("launcher client JSON nesting is invalid") from exc
    return LauncherClientSettings.model_validate_json(value, strict=True)


__all__ = ["LauncherClientSettings", "parse_launcher_client"]
