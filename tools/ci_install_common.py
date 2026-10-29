# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared CI install helpers

"""Helpers for CI installs that need the local engine wheel first."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = ROOT / "bridge"
DIST_DIR = ROOT / "dist"
MATURIN_REQUIREMENTS = ROOT / "requirements" / "maturin.txt"
TRACKED_WINDOWS_EXTENSION = (
    BRIDGE_DIR / "sc_neurocore_engine" / "sc_neurocore_engine.cp312-win_amd64.pyd"
)


def _run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd or ROOT, check=True)


def _sync_engine_extension_into_bridge() -> None:
    """Copy the installed extension into bridge/ for pytest's path injection.

    Root ``conftest.py`` prepends ``bridge/`` to ``sys.path`` so test imports
    resolve the checkout copy of ``sc_neurocore_engine`` before site-packages.
    CI therefore needs the version-matching compiled extension present inside
    ``bridge/sc_neurocore_engine`` as well, not only inside the installed wheel.
    """
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not ext_suffix:
        raise RuntimeError("Could not determine Python extension suffix for CI install")

    site_packages = Path(sysconfig.get_path("purelib"))
    installed_dir = site_packages / "sc_neurocore_engine"
    installed_candidates = sorted(installed_dir.glob(f"sc_neurocore_engine*{ext_suffix}"))
    if not installed_candidates:
        raise FileNotFoundError(
            f"No installed engine extension matching {ext_suffix!r} found in {installed_dir}"
        )

    bridge_pkg_dir = BRIDGE_DIR / "sc_neurocore_engine"
    for stale in bridge_pkg_dir.glob("sc_neurocore_engine*.so"):
        if stale.name != TRACKED_WINDOWS_EXTENSION.name:
            stale.unlink()
    for stale in bridge_pkg_dir.glob("sc_neurocore_engine*.pyd"):
        if stale.name != TRACKED_WINDOWS_EXTENSION.name:
            stale.unlink()

    shutil.copy2(installed_candidates[-1], bridge_pkg_dir / installed_candidates[-1].name)


def install_editable(extra: str) -> int:
    """Install this checkout with its built or same-run shared engine wheel.

    Parameters
    ----------
    extra : str
        Comma-separated project extras to install for the selected CI lane.

    Returns
    -------
    int
        Zero after native import alignment and version checks succeed.
    """
    shared = os.environ.get("SC_NEUROCORE_CI_ENGINE_WHEEL_DIR")
    if shared is None:
        DIST_DIR.mkdir(exist_ok=True)
        if TRACKED_WINDOWS_EXTENSION.exists():
            TRACKED_WINDOWS_EXTENSION.unlink()
        for wheel in DIST_DIR.glob("sc_neurocore_engine-*.whl"):
            wheel.unlink()
        _run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "-r",
            str(MATURIN_REQUIREMENTS),
        )
        _run("maturin", "build", "--release", "--out", str(DIST_DIR), cwd=BRIDGE_DIR)
    wheel_dir = Path(shared) if shared is not None else DIST_DIR
    wheels = sorted(wheel_dir.glob("sc_neurocore_engine-*.whl"))
    if len(wheels) != 1:
        raise FileNotFoundError(f"Expected one engine wheel in {wheel_dir}, found {len(wheels)}")

    _run(sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[0]))
    _sync_engine_extension_into_bridge()
    _run(sys.executable, "-m", "pip", "install", "-e", f".[{extra}]")
    _run(sys.executable, str(ROOT / "tools" / "version_surface_audit.py"))
    return 0
