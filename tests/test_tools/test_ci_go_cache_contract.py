# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go toolchain cache dependency contract

from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_ci_setup_go_caches_tracked_backend_manifests() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    setup_go = next(step for step in steps if step.get("uses", "").startswith("actions/setup-go@"))
    dependency_paths = setup_go["with"]["cache-dependency-path"].splitlines()

    assert dependency_paths == [
        "src/sc_neurocore/accel/go/go.mod",
        "src/sc_neurocore/accel/go/services/hil_debugger/go.sum",
    ]
    assert all((ROOT / relative).is_file() for relative in dependency_paths)
    assert all(relative in _tracked_files() for relative in dependency_paths)


def _tracked_files() -> set[str]:
    """Return repository-relative paths recorded in the Git index."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return set(completed.stdout.split("\0"))
