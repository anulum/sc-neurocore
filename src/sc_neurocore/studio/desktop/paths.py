# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-user desktop data and runtime locations

"""Separate desktop installation resources from durable per-user state."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.project_paths import PROJECT_ROOT_ENV, resolve_studio_project_root


@dataclass(frozen=True, slots=True)
class DesktopPaths:
    """XDG directories and the legacy project location for one desktop user.

    Attributes
    ----------
    config, data, state, cache : pathlib.Path
        Canonical per-user locations outside the application installation.
    legacy_projects : pathlib.Path
        Existing standalone project namespace; never moved automatically.
    """

    config: Path
    data: Path
    state: Path
    cache: Path
    legacy_projects: Path

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> DesktopPaths:
        """Resolve per-user paths without creating directories.

        Parameters
        ----------
        environ : mapping of str to str, optional
            User environment. Empty XDG values use the specified defaults.

        Returns
        -------
        DesktopPaths
            Absolute locations for the selected user profile.

        Raises
        ------
        DesktopStartRefused
            A configured base directory is relative or unresolvable.
        """
        source = os.environ if environ is None else environ
        home = _absolute(source.get("HOME") or str(Path.home()), "HOME")
        roots = [
            _absolute(
                str(Path(source.get(key) or str(home / fallback)) / "sc-neurocore" / "studio"), key
            )
            for key, fallback in (
                ("XDG_CONFIG_HOME", ".config"),
                ("XDG_DATA_HOME", ".local/share"),
                ("XDG_STATE_HOME", ".local/state"),
                ("XDG_CACHE_HOME", ".cache"),
            )
        ]
        config, data, state, cache = roots
        return cls(config, data, state, cache, home / ".sc-neurocore" / "studio" / "projects")

    def prepare(self) -> None:
        """Create private profile directories without touching installation resources.

        Raises
        ------
        OSError
            A location cannot be created as a directory.
        """
        for directory in (self.config, self.data, self.state, self.cache):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def project_root(self, environ: Mapping[str, str], stored: Path | None = None) -> Path:
        """Select an explicit, stored, existing legacy, or new XDG project root.

        Parameters
        ----------
        environ : mapping of str to str
            Process environment; an explicit project override takes precedence.
        stored : pathlib.Path, optional
            Directory retained by a previous successful desktop launch.

        Returns
        -------
        pathlib.Path
            Canonical project namespace. No files are migrated or created.

        Raises
        ------
        DesktopStartRefused
            Existing legacy data cannot be inspected safely.
        StudioProjectRootRefused
            The selected directory is unusable.
        """
        if PROJECT_ROOT_ENV in environ:
            return resolve_studio_project_root(environ=environ)
        if stored is not None:
            return resolve_studio_project_root(environ={}, legacy_root=stored)
        try:
            self.legacy_projects.stat()
        except FileNotFoundError:
            legacy = self.data / "projects"
        except OSError:
            raise DesktopStartRefused(
                "The existing Studio project directory cannot be read."
            ) from None
        else:
            legacy = self.legacy_projects
        return resolve_studio_project_root(environ={}, legacy_root=legacy)


def _absolute(value: str, name: str) -> Path:
    """Validate an XDG or home base before appending application directories."""
    path = Path(value)
    if not path.is_absolute():
        raise DesktopStartRefused(f"{name} must name an absolute directory.")
    try:
        resolved = path.resolve()
        try:
            mode = resolved.stat().st_mode
        except FileNotFoundError:
            return resolved
    except (OSError, RuntimeError, ValueError):
        raise DesktopStartRefused(f"{name} could not be resolved.") from None
    if not stat.S_ISDIR(mode):
        raise DesktopStartRefused(f"{name} must name a directory.")
    return resolved
