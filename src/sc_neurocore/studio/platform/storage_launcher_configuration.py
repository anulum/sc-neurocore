# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher operator configuration

"""Operator-owned configuration of the worker launcher; no request can change it."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

_Positive = Annotated[int, Field(gt=0)]
_Uid = Annotated[int, Field(ge=0, lt=0xFFFFFFFF)]


class LauncherConfiguration(BaseModel):
    """Operator-owned launcher configuration; no request can change it.

    ``python_path`` lists the import roots given to the fixed bootstrap. The
    worker ceilings are passed to the bootstrap, which applies them before
    reading any request data. ``max_records`` bounds retained generation
    records used to answer lost-reply status queries.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    api_uid: _Uid
    worker_uid: Annotated[int, Field(gt=0, lt=0xFFFFFFFF)]
    worker_gid: Annotated[int, Field(gt=0, lt=0xFFFFFFFF)]
    spool_root: Path
    socket_path: Path
    python_executable: Path
    python_path: tuple[Path, ...]
    max_workers: _Positive
    max_records: _Positive
    max_memory_bytes: _Positive
    max_cpu_seconds: _Positive
    max_open_files: _Positive
    max_file_bytes: _Positive
    max_processes: _Positive
    stop_rounds: _Positive
    transfer_timeout_seconds: Annotated[float, Field(gt=0, le=60)]

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Require absolute normalised paths and at least one worker import root.

        Distinct API and compute identities are enforced when a privileged
        launcher starts. An unprivileged launcher may share its identity with
        the API for functional verification only; the API grant still refuses
        any worker whose kernel UID differs from its configured compute UID.
        """
        for path in (self.spool_root, self.socket_path, self.python_executable, *self.python_path):
            if not path.is_absolute() or path != Path(os.path.normpath(path)):
                raise ValueError("launcher paths must be absolute and normalised")
        if not self.python_path:
            raise ValueError("launcher needs at least one worker import root")
        return self


CONFIGURATION_MAX_BYTES = 65536


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate launcher configuration field")
        fields[name] = value
    return fields


def load_launcher_configuration(path: Path) -> LauncherConfiguration:
    """Read the operator configuration file with strict, duplicate-free JSON.

    Parameters
    ----------
    path : Path
        Operator-owned configuration file.

    Returns
    -------
    LauncherConfiguration
        Validated configuration.

    Raises
    ------
    ValueError
        Size, JSON or any field is invalid.
    OSError
        The file cannot be read.
    """
    raw = path.read_bytes()
    if not 0 < len(raw) <= CONFIGURATION_MAX_BYTES:
        raise ValueError("launcher configuration exceeds byte limit")
    try:
        text = raw.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid launcher configuration JSON") from exc
    return LauncherConfiguration.model_validate_json(text, strict=True)
