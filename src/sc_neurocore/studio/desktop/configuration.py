# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Persistent desktop origin and project configuration

"""Retain the browser origin and project namespace across desktop restarts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.project_paths import PROJECT_ROOT_ENV, resolve_studio_project_root

CONFIGURATION_SCHEMA = "studio.desktop.configuration.v1"
MAX_CONFIGURATION_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class DesktopConfiguration:
    """The admitted loopback port and durable project namespace.

    Attributes
    ----------
    port : int
        Reserved listening port, retained to preserve browser storage origin.
    project_root : pathlib.Path
        Absolute project location selected before the backend starts.
    """

    port: int
    project_root: Path


def read_configuration(path: Path) -> DesktopConfiguration | None:
    """Read a bounded configuration without overwriting unknown schema versions.

    Parameters
    ----------
    path : pathlib.Path
        Configuration file in the user's XDG configuration directory.

    Returns
    -------
    DesktopConfiguration or None
        Valid retained configuration, or no prior successful launch.

    Raises
    ------
    DesktopStartRefused
        The file is unreadable, malformed, oversized, or unsupported.
    StudioProjectRootRefused
        The retained project directory is unusable.
    """
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_CONFIGURATION_BYTES + 1)
    except FileNotFoundError:
        return None
    except OSError:
        raise DesktopStartRefused("The desktop configuration cannot be read.") from None
    if len(raw) > MAX_CONFIGURATION_BYTES:
        raise DesktopStartRefused("The desktop configuration exceeds its size limit.")
    try:
        value: object = json.loads(raw)
    except (ValueError, UnicodeError):
        raise DesktopStartRefused("The desktop configuration is not valid JSON.") from None
    if not isinstance(value, dict) or set(value) != {"schema_version", "port", "project_root"}:
        raise DesktopStartRefused("The desktop configuration has unsupported fields.")
    if value["schema_version"] != CONFIGURATION_SCHEMA:
        raise DesktopStartRefused("The desktop configuration version is not supported.")
    port = value["port"]
    project_root = value["project_root"]
    if type(port) is not int or not 1 <= port <= 65_535 or not isinstance(project_root, str):
        raise DesktopStartRefused("The desktop configuration contains invalid values.")
    resolved = resolve_studio_project_root(environ={PROJECT_ROOT_ENV: project_root})
    return DesktopConfiguration(port, resolved)


def write_configuration(path: Path, configuration: DesktopConfiguration) -> None:
    """Atomically retain an admitted origin and namespace with private permissions.

    Parameters
    ----------
    path : pathlib.Path
        File whose parent directory was prepared by the desktop launcher.
    configuration : DesktopConfiguration
        Configuration from a backend that passed readiness admission.

    Raises
    ------
    OSError
        The write or atomic replacement failed. The previous file is preserved
        unless replacement completed.
    """
    payload = {
        "schema_version": CONFIGURATION_SCHEMA,
        "port": configuration.port,
        "project_root": str(configuration.project_root),
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
