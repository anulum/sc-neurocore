# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI Julia runtime export acceptance

"""Exercise real Julia initialization and runner environment file publication."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.storage_event_worker_configuration import EventJuliaRuntime
from tools.studio_event_runtime_environment import export_runtime_environment, main


def test_provisioned_runtime_exports_exact_paths_and_preserves_runner_values(
    tmp_path: Path,
) -> None:
    """The real JuliaCall runtime survives in the next step's environment."""
    destination = tmp_path / "github-env"
    destination.write_text("RETAINED=existing\n", encoding="utf-8")
    assert main(["--github-env", str(destination)]) == 0
    values = dict(line.split("=", 1) for line in destination.read_text().splitlines())
    assert values == {
        "RETAINED": "existing",
        "PYTHON_JULIACALL_EXE": str(Path(os.environ["PYTHON_JULIACALL_EXE"]).resolve()),
        "PYTHON_JULIACALL_PROJECT": str(Path(os.environ["PYTHON_JULIACALL_PROJECT"]).resolve()),
        "PYTHON_JULIACALL_THREADS": "1",
        "PYTHON_JULIACALL_HANDLE_SIGNALS": "no",
    }
    # A subsequent Python process consumes the exported paths, as a CI step does.
    completed = subprocess.run(
        [sys.executable, "-c", "import juliacall; print(juliacall.Main.seval('1+1'))"],
        env={**os.environ, **values},
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    assert completed.stdout.rstrip().endswith("2"), completed.stdout


@pytest.mark.parametrize("separator", ["\n", "\r"])
def test_runtime_export_refuses_environment_injection_before_writing(
    tmp_path: Path, separator: str
) -> None:
    """A real installed project path cannot inject an additional runner variable."""
    project = tmp_path / f"project{separator}INJECTED=value"
    project.symlink_to(Path(os.environ["PYTHON_JULIACALL_PROJECT"]), target_is_directory=True)
    runtime = EventJuliaRuntime(
        executable=Path(os.environ["PYTHON_JULIACALL_EXE"]),
        project=project,
        handle_signals="no",
    )
    destination = tmp_path / "github-env"
    destination.write_bytes(b"RETAINED=existing\n")
    with pytest.raises(ValueError, match="line separators"):
        export_runtime_environment(runtime, destination)
    assert destination.read_bytes() == b"RETAINED=existing\n"


def test_runtime_export_refuses_unwritable_destination(tmp_path: Path) -> None:
    """A runner file error is propagated without claiming successful provisioning."""
    runtime = EventJuliaRuntime(
        executable=Path(os.environ["PYTHON_JULIACALL_EXE"]),
        project=Path(os.environ["PYTHON_JULIACALL_PROJECT"]),
        handle_signals="no",
    )
    with pytest.raises(IsADirectoryError):
        export_runtime_environment(runtime, tmp_path)


def test_runtime_export_cli(tmp_path: Path) -> None:
    """The workflow's executable tool exports a usable explicit runtime."""
    tool = Path(__file__).resolve().parents[1] / "tools/studio_event_runtime_environment.py"
    destination = tmp_path / "github-env"
    subprocess.run(
        [sys.executable, str(tool), "--github-env", str(destination)],
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert "PYTHON_JULIACALL_PROJECT=" in destination.read_text()
