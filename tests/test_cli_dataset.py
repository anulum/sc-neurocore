# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The dataset command records, verifies and splits real files

"""``sc-neurocore dataset`` from manifest to verified split, through the public CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.datasets.manifest import manifest_from_dict
from sc_neurocore.datasets.splits import leaked_groups, split_plan_from_dict
from tests.cli_test_support import run_cli
from tests.event_dataset_support import write_nmnist, write_shd


def test_manifest_verify_and_split_run_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "shd"
    write_shd(data, {"train": [1, 1, 2, 3, 4, 4, 5], "test": [5, 9]})
    manifest_path = tmp_path / "shd.manifest.json"

    assert (
        run_cli(
            "dataset", "manifest", "shd", str(data), "--version", "1.0", "-o", str(manifest_path)
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Spiking Heidelberg Digits 1.0: 2 files" in output
    assert "train: 7 samples in 5 groups" in output
    assert "licence: CC-BY-4.0" in output
    assert "published splits share 1 groups: speaker:5" in output
    manifest = manifest_from_dict(json.loads(manifest_path.read_text()))
    assert manifest.digest in output

    assert run_cli("dataset", "verify", str(manifest_path), str(data)) == 0
    assert f"2 files match {manifest.digest}" in capsys.readouterr().out

    plan_path = tmp_path / "plan.json"
    arguments = ["--part", "train=0.75", "--part", "validation=0.25", "--seed", "4"]
    assert run_cli("dataset", "split", str(manifest_path), *arguments, "-o", str(plan_path)) == 0
    output = capsys.readouterr().out
    plan = split_plan_from_dict(json.loads(plan_path.read_text()))
    assert plan.digest in output
    assert leaked_groups(manifest, plan) == ()
    assert f"validation: {len(plan.assignment['validation'])} samples" in output


def test_verify_reports_a_changed_file_with_status_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_nmnist(tmp_path / "nmnist", {"train": {0: 2}})
    manifest_path = tmp_path / "m.json"
    run_cli(
        "dataset",
        "manifest",
        "nmnist",
        str(tmp_path / "nmnist"),
        "--version",
        "1",
        "-o",
        str(manifest_path),
    )
    capsys.readouterr()
    (tmp_path / "nmnist/Train/0/00001.bin").write_bytes(b"\x00" * 5)

    assert run_cli("dataset", "verify", str(manifest_path), str(tmp_path / "nmnist")) == 1
    assert capsys.readouterr().out == "changed: Train/0/00001.bin\n"


@pytest.mark.parametrize(
    "argv",
    [
        ["dataset", "manifest", "nmnist", "{empty}", "--version", "1", "-o", "{out}"],
        ["dataset", "verify", "{missing}", "{empty}"],
        ["dataset", "split", "{manifest}", "--part", "train", "-o", "{out}"],
        ["dataset", "split", "{manifest}", "--part", "a=0.5", "--part", "a=0.5", "-o", "{out}"],
        ["dataset", "split", "{manifest}", "--part", "a=half", "--part", "b=0.5", "-o", "{out}"],
        [
            "dataset",
            "split",
            "{manifest}",
            "--part",
            "a=0.5",
            "--part",
            "b=0.5",
            "--source",
            "x",
            "-o",
            "{out}",
        ],
    ],
)
def test_a_refused_request_prints_why_and_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    write_nmnist(tmp_path / "data", {"train": {0: 2}})
    manifest = tmp_path / "m.json"
    run_cli(
        "dataset",
        "manifest",
        "nmnist",
        str(tmp_path / "data"),
        "--version",
        "1",
        "-o",
        str(manifest),
    )
    capsys.readouterr()
    (tmp_path / "empty").mkdir()
    paths = {
        "empty": str(tmp_path / "empty"),
        "out": str(tmp_path / "out.json"),
        "missing": str(tmp_path / "missing.json"),
        "manifest": str(manifest),
    }
    assert run_cli(*[part.format(**paths) for part in argv]) == 2
    assert capsys.readouterr().out.startswith("Error: ")
    assert not (tmp_path / "out.json").exists()


def test_the_installed_entry_point_runs_the_command(tmp_path: Path) -> None:
    write_nmnist(tmp_path / "data", {"train": {0: 1}, "test": {1: 1}})
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "sc_neurocore.cli",
            "dataset",
            "manifest",
            "nmnist",
            str(tmp_path / "data"),
            "--version",
            "1",
            "-o",
            str(tmp_path / "m.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "N-MNIST 1: 2 files" in run.stdout
    assert "licence: CC-BY-SA-4.0" in run.stdout
