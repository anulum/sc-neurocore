# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio scratch job root lifecycle tests

"""A Studio without a configured job root removes its scratch root when its process exits.

An unconfigured job root is scratch, private to the process that made it: no later
process reconciles it. These tests build the Studio app in a real child process with
a private temporary directory and look at that directory after the child has exited.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_PREFIX = "sc-neurocore-studio-jobs-"

#: Builds the Studio app and prints how many scratch roots the temporary directory holds.
CHILD = (
    "import pathlib, tempfile\n"
    "from sc_neurocore.studio.app import create_app\n"
    "create_app()\n"
    f"print(len(list(pathlib.Path(tempfile.gettempdir()).glob('{SCRATCH_PREFIX}*'))))\n"
)


def _run_child(temporary: Path, job_root: Path | None) -> subprocess.CompletedProcess[str]:
    """Build the Studio app in a child process whose temporary directory is ``temporary``.

    Parameters
    ----------
    temporary : Path
        The child's temporary directory (``TMPDIR``).
    job_root : Path or None
        The configured job root, or ``None`` for none.

    Returns
    -------
    subprocess.CompletedProcess
        The child's exit status and output: the number of scratch roots while it ran.
    """
    env = {key: value for key, value in os.environ.items() if key != "SC_NEUROCORE_STUDIO_JOB_ROOT"}
    env["TMPDIR"] = str(temporary)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    if job_root is not None:
        env["SC_NEUROCORE_STUDIO_JOB_ROOT"] = str(job_root)
    return subprocess.run(
        [sys.executable, "-c", CHILD],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _scratch_roots(temporary: Path) -> list[Path]:
    """Return the scratch job roots in a temporary directory.

    Parameters
    ----------
    temporary : Path
        The directory to look in.

    Returns
    -------
    list of Path
        The scratch roots it holds.
    """
    return sorted(temporary.glob(f"{SCRATCH_PREFIX}*"))


def test_an_unconfigured_root_is_removed_when_its_process_exits(tmp_path: Path) -> None:
    """The scratch root exists while its process runs and is gone after it exits."""
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    result = _run_child(temporary, job_root=None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "1"
    assert _scratch_roots(temporary) == []


def test_a_configured_root_is_kept_when_its_process_exits(tmp_path: Path) -> None:
    """A configured root is durable: it outlives the process, and no scratch root is made."""
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    job_root = tmp_path / "jobs"
    result = _run_child(temporary, job_root=job_root)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "0"
    assert job_root.is_dir()
    assert any(job_root.iterdir())
    assert _scratch_roots(temporary) == []


def test_only_the_process_that_made_a_scratch_root_removes_it(tmp_path: Path) -> None:
    """A forked child inherits the exit handler; the guard makes it leave the root alone."""
    from sc_neurocore.studio.api.runtime import _remove_scratch_job_root

    root = tmp_path / f"{SCRATCH_PREFIX}guarded"
    root.mkdir()
    (root / "ledger").write_text("records", encoding="utf-8")
    _remove_scratch_job_root(root, owner_pid=os.getpid() + 1)
    assert (root / "ledger").is_file()
    _remove_scratch_job_root(root, owner_pid=os.getpid())
    assert not root.exists()
