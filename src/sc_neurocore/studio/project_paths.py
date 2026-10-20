# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio project directory configuration

"""Resolve the same project namespace for persistence and recovery planning."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path

from sc_neurocore.refusals import AuthoredRefusal

PROJECT_ROOT_ENV = "SC_NEUROCORE_STUDIO_PROJECT_ROOT"


class StudioProjectRootRefused(AuthoredRefusal):
    """A deliberate refusal of an unusable process-wide project directory."""


def resolve_studio_project_root(
    *, environ: Mapping[str, str] | None = None, legacy_root: Path | None = None
) -> Path:
    """Resolve the configured project directory without creating or moving data.

    Parameters
    ----------
    environ : mapping of str to str, optional
        Process configuration. A configured ``SC_NEUROCORE_STUDIO_PROJECT_ROOT``
        must be a nonempty absolute path, set before starting the application.
        When omitted, read the actual process environment.
    legacy_root : pathlib.Path, optional
        Existing project location when the override is absent. The default is
        ``~/.sc-neurocore/studio/projects``. Keeping this fallback preserves
        existing user data; selecting another root never migrates it silently.

    Returns
    -------
    pathlib.Path
        Absolute canonical directory shared by project I/O and backup planning.

    Raises
    ------
    StudioProjectRootRefused
        The configured value is empty, relative, unresolvable, or an existing
        filesystem object that is not a directory.
    """
    source = os.environ if environ is None else environ
    configured = source.get(PROJECT_ROOT_ENV)
    if configured is not None:
        if not configured.strip():
            raise StudioProjectRootRefused("Studio project root must be a nonempty absolute path.")
        root = Path(configured)
        if not root.is_absolute():
            raise StudioProjectRootRefused("Studio project root must be a nonempty absolute path.")
    else:
        root = (
            Path.home() / ".sc-neurocore" / "studio" / "projects"
            if legacy_root is None
            else legacy_root
        )
    try:
        resolved = root.expanduser().resolve()
        try:
            mode = resolved.stat().st_mode
        except FileNotFoundError:
            return resolved
    except (OSError, RuntimeError, ValueError):
        raise StudioProjectRootRefused("Studio project root could not be resolved.") from None
    if not stat.S_ISDIR(mode):
        raise StudioProjectRootRefused("Studio project root must name a directory.")
    return resolved
