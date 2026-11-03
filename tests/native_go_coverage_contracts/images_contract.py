# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual native image and observer contracts

"""Damage declarations of real current images without fabricating native providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tools.go_coverage_images import read_images
from tools.go_mat_coverage_build import NativeGo


def test_complete_declarations_bind_four_actual_current_images(native_job: Path) -> None:
    """All four roles resolve to real native images consumed by the original ABI tests."""
    images = read_images(native_job / "images.json", native_job)
    assert {(image.backend, image.model) for image in images.values()} == {
        (backend, model) for backend in ("go", "mojo") for model in ("mat", "sc_resetting_mat")
    }
    assert len(images) == 4 and all(image.path.is_file() for image in images.values())


def test_real_bytes_path_load_retains_native_snapshots(
    native_job: Path, native_builder: NativeGo
) -> None:
    """An actual ctypes bytes filename cannot bypass the real image observer."""
    environment = json.loads((native_job / "environments/contracts.json").read_text())
    driver = native_builder.job / "artifacts/bytes-load-driver.py"
    driver.write_text(
        "import ctypes,os,subprocess,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from tools.go_coverage_images import register,read_images;"
        "job=Path(sys.argv[2]);register(job,'contracts');ctypes.CDLL(None);"
        "ctypes.CDLL('libc.so.6');"
        "subprocess.run([sys.executable,'-I','-B','-c','pass'],check=True);"
        "image=next(i for i in read_images(job/'images.json',job).values() "
        "if i.backend=='go' and i.model=='mat');ctypes.CDLL(os.fsencode(image.path))\n"
    )
    assert (
        native_builder.command(
            "bytes-load",
            native_builder.python_script(driver, [str(native_builder.repository), str(native_job)]),
            native_builder.job,
            environment=environment,
        )
        == 0
    )
    receipt = json.loads((native_builder.job / "commands/bytes-load/receipt.json").read_text())
    rows = [
        json.loads(line)
        for line in (
            native_job / "process-counters/contracts" / str(receipt["pid"]) / "observations.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert rows[-1]["event"] == "finished"
    assert {row["capture"] for row in rows if row["event"] == "snapshot"} == {"first", "final"}


def test_actual_snapshot_directory_collision_exits_without_success(
    native_job: Path, native_builder: NativeGo
) -> None:
    """A real directory conflict stops observer exit with 97 and preserves its trace."""
    environment = json.loads((native_job / "environments/contracts.json").read_text())
    driver = native_builder.job / "artifacts/snapshot-collision-driver.py"
    driver.write_text(
        "import atexit,ctypes,os,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from tools.go_coverage_images import register,read_images;"
        "job=Path(sys.argv[2]);register(job,'contracts');"
        "image=next(i for i in read_images(job/'images.json',job).values() "
        "if i.backend=='go' and i.model=='mat');ctypes.CDLL(str(image.path));"
        "(job/'process-counters'/'contracts'/str(os.getpid())/image.model/'first')"
        ".mkdir(parents=True);atexit._run_exitfuncs()\n"
    )
    assert (
        native_builder.command(
            "snapshot-collision",
            native_builder.python_script(driver, [str(native_builder.repository), str(native_job)]),
            native_builder.job,
            environment=environment,
        )
        == 97
    )
    receipt = json.loads(
        (native_builder.job / "commands/snapshot-collision/receipt.json").read_text()
    )
    rows = [
        json.loads(line)
        for line in (
            native_job / "process-counters/contracts" / str(receipt["pid"]) / "observations.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert rows[-1]["event"] == "snapshot_failure"
    assert "FileExistsError" in rows[-1]["error"]
    assert any(row["event"] == "invalid_snapshot_path" and row["status"] != 0 for row in rows)
    assert not any(row["event"] == "finished" for row in rows)
    counters = [
        path
        for path in native_builder.python_coverage_config.parent.glob(".coverage.*")
        if f".pid{receipt['pid']}." in path.name
    ]
    assert len(counters) == 1 and counters[0].stat().st_size > 0


def test_actual_loader_refuses_an_unadmitted_owned_library_copy(
    native_job: Path,
    native_builder: NativeGo,
) -> None:
    """The real ctypes audit hook blocks a genuine file outside its declared image cohort."""
    import shutil

    images = read_images(native_job / "images.json", native_job)
    image = next(i for i in images.values() if i.backend == "go")
    copy = native_builder.job / "artifacts" / image.path.name
    shutil.copyfile(image.path, copy)
    driver = native_builder.job / "artifacts/unadmitted-load-driver.py"
    driver.write_text(
        "import ctypes,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from tools.go_coverage_images import register;"
        "register(Path(sys.argv[2]),'contracts');ctypes.CDLL(sys.argv[3])\n"
    )
    environment = json.loads((native_job / "environments/contracts.json").read_text())
    assert (
        native_builder.command(
            "unadmitted-load",
            native_builder.python_script(
                driver, [str(native_builder.repository), str(native_job), str(copy)]
            ),
            native_builder.job,
            environment=environment,
        )
        == 1
    )
    assert (
        "Unadmitted MAT family image"
        in (native_builder.job / "commands/unadmitted-load/stderr.log").read_text()
    )


@pytest.mark.parametrize(
    "fault",
    [
        "count",
        "entry",
        "field",
        "type",
        "relative",
        "alias",
        "foreign",
        "digest",
        "role",
        "duplicate",
    ],
)
def test_damaged_actual_image_declarations_refuse(
    native_job: Path,
    native_repository: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    """A damaged copy of the genuine producer manifest cannot change the admitted cohort."""
    raw = json.loads((native_job / "images.json").read_text())
    rows = cast(list[dict[str, object]], raw)
    if fault == "count":
        rows.pop()
    elif fault == "entry":
        raw[0] = "not a declaration"
    elif fault == "field":
        rows[0]["unknown"] = "value"
    elif fault == "type":
        rows[0]["path"] = 1
    elif fault == "relative":
        rows[0]["path"] = "relative/libmat.so"
    elif fault == "alias":
        alias = tmp_path / "alias.so"
        alias.symlink_to(str(rows[0]["path"]))
        rows[0]["path"] = str(alias)
    elif fault == "foreign":
        rows[0]["path"] = str(native_repository / "src/sc_neurocore/accel/mojo/mat/libmat.so")
    elif fault == "digest":
        rows[0]["sha256"] = "not a digest"
    elif fault == "role":
        rows[0]["model"] = "another-model"
    else:
        rows[1] = dict(rows[0])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        read_images(path, native_job)


def test_actual_closed_registration_is_idempotent(
    native_job: Path, native_builder: NativeGo
) -> None:
    """Two real public registrations attach one observer to the actual current process."""
    environment = json.loads((native_job / "environments/contracts.json").read_text())
    code = (
        "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from tools.go_coverage_images import register;"
        "register(Path(sys.argv[2]),'contracts');register(Path(sys.argv[2]),'contracts')"
    )
    driver = native_builder.job / "artifacts/registration-driver.py"
    driver.write_text(code + "\n")
    assert (
        native_builder.command(
            "register-real",
            native_builder.python_script(
                driver,
                [
                    str(native_builder.repository),
                    str(native_job),
                ],
            ),
            native_builder.job,
            environment=environment,
        )
        == 0
    )
    receipt = json.loads((native_builder.job / "commands/register-real/receipt.json").read_text())
    rows = [
        json.loads(line)
        for line in (
            native_job / "process-counters/contracts" / str(receipt["pid"]) / "observations.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert [row["event"] for row in rows] == ["registered", "closed_environment", "finished"]


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("optimized", "unoptimized Python"),
        ("phase", "Unknown native coverage"),
        ("manifest", "Unknown native coverage"),
        ("required-type", "must map strings to strings"),
        ("unreviewed-key", "unreviewed consumer environment key"),
        ("actual-key", "Actual consumer keys or values differ"),
        ("job-identity", "Consumer task, phase or manifest identity differs"),
        ("phase-identity", "Consumer task, phase or manifest identity differs"),
        ("manifest-identity", "Consumer task, phase or manifest identity differs"),
        ("interpreter", "Actual Python executable or prefix differs"),
    ],
)
def test_real_registration_refuses_damaged_closed_inputs_before_loading(
    native_job: Path,
    native_builder: NativeGo,
    tmp_path: Path,
    fault: str,
    reason: str,
) -> None:
    """Real subprocesses reject changed protocol inputs before any declared native load."""
    job = tmp_path / "registration-input"
    (job / "environments").mkdir(parents=True)
    expected = json.loads((native_job / "environments/contracts.json").read_text())
    expected["NEUROCORE_GO_COVERAGE_JOB"] = str(job)
    interpreter = json.loads((native_job / "python.json").read_text())
    phase, manifest = "contracts", "images.json"
    if fault == "phase":
        phase = "unknown"
    elif fault == "manifest":
        manifest = "unknown.json"
    elif fault == "unreviewed-key":
        expected["LD_LIBRARY_PATH"] = "/absent"
    elif fault == "job-identity":
        expected["NEUROCORE_GO_COVERAGE_JOB"] = str(job.parent)
    elif fault == "phase-identity":
        expected["NEUROCORE_GO_COVERAGE_PHASE"] = "before"
    elif fault == "manifest-identity":
        expected["NEUROCORE_GO_COVERAGE_MANIFEST"] = "images-normal.json"
    elif fault == "interpreter":
        interpreter["prefix"] += "/different"
    actual = dict(expected)
    if fault == "actual-key":
        actual["GODEBUG"] = "panicnil=1"
    elif fault == "required-type":
        expected["HOME"] = 1
    (job / "environments/contracts.json").write_text(json.dumps(expected))
    (job / "python.json").write_text(json.dumps(interpreter))
    code = (
        "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from tools.go_coverage_images import register;"
        "register(Path(sys.argv[2]),sys.argv[3],sys.argv[4])"
    )
    driver = native_builder.job / "artifacts/registration-driver.py"
    driver.write_text(code + "\n")
    args = native_builder.python_script(
        driver,
        [
            str(native_builder.repository),
            str(job),
            phase,
            manifest,
        ],
        optimized=fault == "optimized",
    )
    assert (
        native_builder.command("register-refusal", args, native_builder.job, environment=actual)
        == 1
    )
    error = (native_builder.job / "commands/register-refusal/stderr.log").read_text()
    assert reason in error
    assert not (job / "process-counters").exists()
