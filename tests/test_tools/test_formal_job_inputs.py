# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual SymbiYosys source-directory contracts

"""Exercise the real SBY parser, checkout inputs and public invocation command."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.formal_job_inputs import main, resolve_job

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "hdl/formal/catalogue"
JOB = "sc_mccullochpittsneuron"


def _copy_job(root: Path) -> Path:
    """Copy the real combinatorial catalogue job and both of its unchanged sources."""
    directory = root / "hdl/formal/catalogue"
    directory.mkdir(parents=True)
    for name in (JOB + ".sby", JOB + ".v", JOB + "_formal.v"):
        shutil.copyfile(CATALOGUE / name, directory / name)
    return directory / (JOB + ".sby")


def test_every_tracked_job_has_one_real_source_convention() -> None:
    """Discover every tracked job, including all five checkout-relative contracts."""
    jobs = subprocess.check_output(
        ["git", "ls-files", "--", "hdl/formal/*.sby"], cwd=ROOT, text=True
    ).splitlines()
    assert jobs
    root_relative = set()
    for job in jobs:
        inputs = resolve_job(ROOT, Path(job))
        assert inputs.job in inputs.sources
        assert len(inputs.sources) >= 2
        assert all(source.is_file() and source.is_relative_to(ROOT) for source in inputs.sources)
        if inputs.directory == ROOT:
            root_relative.add(job)
    assert {
        "hdl/formal/sc_aer_priority_queue.sby",
        "hdl/formal/sc_dcls_layer_core.sby",
        "hdl/formal/sensors/adc_to_spike_quantiser.sby",
        "hdl/formal/timing/example_cdc_two_flop_synchroniser.sby",
        "hdl/formal/timing/example_dense_layer_core_latency.sby",
    } <= root_relative


@pytest.mark.parametrize(
    "job,directory,argument",
    [
        ("hdl/formal/edt_controller.sby", "hdl/formal", "edt_controller.sby"),
        (
            "hdl/formal/sensors/adc_to_spike_quantiser.sby",
            ".",
            "hdl/formal/sensors/adc_to_spike_quantiser.sby",
        ),
    ],
)
def test_cli_paths_preserve_both_existing_invocations(
    job: str, directory: str, argument: str
) -> None:
    """The actual command emits two quoted-shell-safe path lines for either convention."""
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "tools/formal_job_inputs.py"), job],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=40,
    )
    assert result.stdout.splitlines() == [directory, argument]


def test_json_binds_the_configuration_and_every_actual_copied_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Changing a copied RTL byte changes its public source manifest and keeps the job."""
    job = _copy_job(tmp_path)
    assert main([str(job), "--repository", str(tmp_path), "--json"]) == 0
    original = json.loads(capsys.readouterr().out)
    expected = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in job.parent.iterdir()
    }
    assert original["sources"] == expected
    dut = job.with_suffix(".v")
    dut.write_bytes(dut.read_bytes() + b"\n// source-identity negative control\n")
    assert main([str(job), "--repository", str(tmp_path), "--json"]) == 0
    changed = json.loads(capsys.readouterr().out)
    key = dut.relative_to(tmp_path).as_posix()
    assert changed["sources"][key] != original["sources"][key]
    assert (
        changed["sources"][job.relative_to(tmp_path).as_posix()]
        == expected[job.relative_to(tmp_path).as_posix()]
    )


def test_root_level_job_has_no_duplicate_directory_candidate(tmp_path: Path) -> None:
    """A real job copied directly to the checkout root remains unambiguous."""
    for name in (JOB + ".sby", JOB + ".v", JOB + "_formal.v"):
        shutil.copyfile(CATALOGUE / name, tmp_path / name)
    inputs = resolve_job(tmp_path, Path(JOB + ".sby"))
    assert inputs.directory == tmp_path


def test_task_conditioned_sources_are_all_in_the_manifest(tmp_path: Path) -> None:
    """Include each task's actual RTL, even when the source is not shared by all tasks."""
    job = _copy_job(tmp_path)
    other = job.parent / "alternate.v"
    shutil.copyfile(job.with_suffix(".v"), other)
    original = job.read_text()
    job.write_text(
        "[tasks]\nfirst\nsecond\n\n"
        + original.replace(
            "\n" + JOB + ".v\n",
            "\nfirst: " + JOB + ".v\nsecond: alternate.v\n",
        )
    )
    inputs = resolve_job(tmp_path, job)
    assert set(inputs.sources) == {
        job,
        job.with_suffix(".v"),
        other,
        job.parent / (JOB + "_formal.v"),
    }


@pytest.mark.parametrize("invalid", ["missing.sby", "README.md", "../outside.sby"])
def test_invalid_job_paths_fail_before_running_the_parser(tmp_path: Path, invalid: str) -> None:
    """Missing jobs, wrong extensions and paths outside the checkout are refused."""
    with pytest.raises(ValueError, match="existing .sby file inside"):
        resolve_job(tmp_path, Path(invalid))


@pytest.mark.parametrize("defect", ["missing", "outside", "ambiguous", "empty", "syntax"])
def test_real_parser_and_input_failures_reject_the_job(tmp_path: Path, defect: str) -> None:
    """Damage original source custody or syntax without substituting a fake SBY executable."""
    job = _copy_job(tmp_path)
    if defect == "missing":
        job.write_text(job.read_text().replace(JOB + ".v", JOB + ".missing.v"))
    elif defect == "outside":
        job.write_text(job.read_text().replace(JOB + ".v", str(CATALOGUE / (JOB + ".v"))))
    elif defect == "ambiguous":
        for name in (JOB + ".v", JOB + "_formal.v"):
            shutil.copyfile(CATALOGUE / name, tmp_path / name)
    elif defect == "empty":
        job.write_text(job.read_text().split("[files]", 1)[0] + "[files]\n")
    else:
        job.write_text(job.read_text() + "\n[tasks]\ninvalid: syntax: extra\n")
    with pytest.raises((ValueError, subprocess.SubprocessError)):
        resolve_job(tmp_path, job)


def test_cli_refuses_invalid_inputs_with_a_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public entry point reports failure instead of emitting usable invocation paths."""
    assert main(["missing.sby", "--repository", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "existing .sby file inside" in captured.err


def test_cli_requires_the_actual_sby_executable(tmp_path: Path) -> None:
    """An unavailable native parser is a real command failure, never a guessed directory."""
    job = _copy_job(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "tools/formal_job_inputs.py"),
            str(job),
            "--repository",
            str(tmp_path),
        ],
        env={**os.environ, "PATH": str(tmp_path / "unavailable-tools")},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "sby" in result.stderr
