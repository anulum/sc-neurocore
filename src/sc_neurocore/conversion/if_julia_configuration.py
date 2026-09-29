# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia owned replay configuration admission

"""Admit explicit offline Julia settings without invoking automatic dependency resolution."""

import importlib.metadata
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

Fingerprint = tuple[object, ...]

_OPTIONS = ("exe", "project", "threads", "handle-signals", "init")
"""JuliaCall startup options the admission compares."""

_SETTINGS = (
    *("PYTHON_JULIACALL_" + name.upper().replace("-", "_") for name in _OPTIONS),
    "JULIA_CONDAPKG_BACKEND",
    "JULIA_PKG_OFFLINE",
)
"""Environment settings the admission reads."""

_ADMITTED: dict[Fingerprint, tuple[str, str]] = {}
"""Successful admissions keyed by the complete configuration they read."""


def _fingerprint() -> Fingerprint | None:
    """Identify every input of an admission: settings, options and file identities.

    Returns
    -------
    tuple or None
        The environment values and JuliaCall ``-X`` options the admission reads and the device,
        inode, size and modification time of the executable and both project
        files; None when any file cannot be examined.
    """
    settings = tuple(os.environ.get(name) for name in _SETTINGS)
    options = tuple(sys._xoptions.get("juliacall-" + name) for name in _OPTIONS)
    project = Path(os.environ.get("PYTHON_JULIACALL_PROJECT", ""))
    files = []
    for path in (
        Path(os.environ.get("PYTHON_JULIACALL_EXE", "")),
        project / "Project.toml",
        project / "Manifest.toml",
    ):
        try:
            state = path.stat()
        except OSError:
            return None
        files.append((state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns))
    return settings, options, tuple(files)


def julia_configuration(user: str = "Julia IF replay") -> tuple[str, str]:
    """Admit explicit matching Julia runtime options before importing JuliaCall.

    Parameters
    ----------
    user : str
        Runtime user named in every refusal, such as ``"Julia QCFS"``.

    Returns
    -------
    tuple of str
        Resolved installed executable and locked project paths.

    Raises
    ------
    RuntimeError
        Options conflict, dependencies differ or required offline settings are absent.

    Notes
    -----
    A successful admission is reused while every setting, option and file
    identity it read is unchanged, so each managed call does not reparse the
    locked project; any change repeats the complete admission.
    """
    key = _fingerprint()
    if key is None:
        return _admit(user)
    if key not in _ADMITTED:
        _ADMITTED[key] = _admit(user)
    return _ADMITTED[key]


def _admit(user: str) -> tuple[str, str]:
    """Run the complete admission that ``julia_configuration`` documents."""
    executable = Path(os.environ.get("PYTHON_JULIACALL_EXE", ""))
    project = Path(os.environ.get("PYTHON_JULIACALL_PROJECT", ""))
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError(f"{user} requires an existing absolute Julia executable")
    if (
        not project.is_absolute()
        or not (project / "Project.toml").is_file()
        or not (project / "Manifest.toml").is_file()
    ):
        raise RuntimeError(f"{user} requires an existing absolute locked Julia project")
    if (
        os.environ.get("PYTHON_JULIACALL_THREADS") != "1"
        or os.environ.get("PYTHON_JULIACALL_HANDLE_SIGNALS") != "yes"
    ):
        raise RuntimeError(f"{user} requires one thread and explicit managed signal handling")
    if (
        os.environ.get("JULIA_CONDAPKG_BACKEND") != "Null"
        or os.environ.get("JULIA_PKG_OFFLINE") != "true"
    ):
        raise RuntimeError(f"{user} requires offline packages and the Null CondaPkg backend")
    try:
        version = importlib.metadata.version("juliacall")
        declaration = tomllib.loads((project / "Project.toml").read_text())
        manifest = tomllib.loads((project / "Manifest.toml").read_text())
        if (
            declaration.get("compat", {}).get("PythonCall") != "=" + version
            or manifest.get("deps", {}).get("PythonCall", [{}])[0].get("version") != version
        ):
            raise RuntimeError(f"{user}: PythonCall and JuliaCall versions must match exactly")
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        raise RuntimeError(f"{user}: locked project or JuliaCall dependency unavailable") from error
    for name, expected in (
        ("exe", str(executable)),
        ("project", str(project)),
        ("threads", "1"),
        ("handle-signals", "yes"),
        ("init", "yes"),
    ):
        effective = sys._xoptions.get(
            "juliacall-" + name,
            os.environ.get("PYTHON_JULIACALL_" + name.upper().replace("-", "_"), expected),
        )
        if effective != expected:
            raise RuntimeError(f"{user}: Python -X or environment startup options conflict")
    return str(executable.resolve()), str(project.resolve())
