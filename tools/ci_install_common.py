# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared CI install helpers

"""Helpers for CI installs that need the local engine wheel first."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_DIR = ROOT / "bridge"
DIST_DIR = ROOT / "dist"
MATURIN_REQUIREMENTS = ROOT / "requirements" / "maturin.txt"
TRACKED_WINDOWS_EXTENSION = (
    BRIDGE_DIR / "sc_neurocore_engine" / "sc_neurocore_engine.cp312-win_amd64.pyd"
)


def _run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd or ROOT, check=True)


def install_editable(extra: str) -> int:
    DIST_DIR.mkdir(exist_ok=True)
    if TRACKED_WINDOWS_EXTENSION.exists():
        TRACKED_WINDOWS_EXTENSION.unlink()
    for wheel in DIST_DIR.glob("sc_neurocore_engine-*.whl"):
        wheel.unlink()
    _run(
        sys.executable, "-m", "pip", "install", "--require-hashes", "-r", str(MATURIN_REQUIREMENTS)
    )
    _run("maturin", "build", "--release", "--out", str(DIST_DIR), cwd=BRIDGE_DIR)

    wheels = sorted(DIST_DIR.glob("sc_neurocore_engine-*.whl"))
    if not wheels:
        raise FileNotFoundError("No sc_neurocore_engine wheel was produced in dist/")

    _run(sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[-1]))
    _run(sys.executable, "-m", "pip", "install", "-e", f".[{extra}]")
    return 0
