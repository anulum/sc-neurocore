# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public native gate inputs and source custody contracts

"""Check the real public entry point's productive refusals without replaying model workloads."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from tools.go_mat_coverage import python_environment, source_inputs
from tools.go_mat_coverage_build import NativeGo


@pytest.mark.parametrize(
    "fault", ["seconds", "jobs", "optimized", "internal-output", "existing-output", "go"]
)
def test_public_gate_refuses_invalid_inputs_without_native_execution(
    native_builder: NativeGo,
    tmp_path: Path,
    fault: str,
) -> None:
    """Actual CLI refusal leaves no child-command allocation or source-side output."""
    output = tmp_path / "refused-gate"
    if fault == "internal-output":
        output = native_builder.repository / ("never-created-native-gate-" + tmp_path.name)
    elif fault == "existing-output":
        output.mkdir()
    go = native_builder.go if fault != "go" else tmp_path / "absent-go"
    command = native_builder.python_script(
        native_builder.repository / "tools/go_mat_coverage.py",
        [
            "--repository",
            str(native_builder.repository),
            "--output",
            str(output),
            "--go",
            str(go),
            "--seconds",
            "0" if fault == "seconds" else "1",
            "--jobs",
            "0" if fault == "jobs" else str(native_builder.jobs),
            "--stop-seconds",
            str(native_builder.stop_seconds),
            "--python-coverage-config",
            str(native_builder.python_coverage_config),
        ],
        optimized=fault == "optimized",
    )
    assert native_builder.command("public-refusal", command, native_builder.job) == 1
    error = (native_builder.job / "commands/public-refusal/stderr.log").read_text()
    assert "Native Go MAT gate refused:" in error
    assert not (output / "commands").exists()
    if fault == "existing-output":
        assert list(output.iterdir()) == []
    else:
        assert not output.exists()


def test_gate_environment_is_explicit_and_cannot_be_redeclared(
    native_repository: Path,
    tmp_path: Path,
    native_builder: NativeGo,
) -> None:
    """A current declared mapping remains exact, while a modified declaration is refused."""
    (tmp_path / "environments").mkdir()
    expected = python_environment(
        tmp_path, native_repository, "contracts", "images.json", native_builder.jobs
    )
    assert expected["PYTHONPATH"] == str(tmp_path / "bootstrap") + ":" + str(native_repository)
    assert expected["NEUROCORE_GO_COVERAGE_JOB"] == str(tmp_path)
    assert not any(key.startswith("LD_") or key == "GODEBUG" for key in expected)
    assert (
        python_environment(
            tmp_path, native_repository, "contracts", "images.json", native_builder.jobs
        )
        == expected
    )
    assert expected["GOMAXPROCS"] == str(native_builder.jobs)
    with pytest.raises(ValueError, match="concurrency"):
        python_environment(tmp_path, native_repository, "contracts", "images.json", 0)
    path = tmp_path / "environments/contracts.json"
    changed = {**expected, "GODEBUG": "panicnil=1"}
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match="previously declared"):
        python_environment(
            tmp_path, native_repository, "contracts", "images.json", native_builder.jobs
        )
    assert json.loads(path.read_text()) == changed


def test_real_source_manifest_binds_current_producer_and_consumer_inputs(
    native_job: Path,
    native_builder: NativeGo,
) -> None:
    """Keep actual first-party source and real producer/consumer executables pinned."""
    actual = source_inputs(
        native_builder.repository, native_builder.go, native_builder.python_coverage_config
    )
    original = json.loads((native_job / "source-before.json").read_text())
    assert actual == original
    for relative in (
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "requirements/build.txt",
        "tools/go_mat_coverage.py",
        "tools/go_coverage_images.py",
        "tests/native_go_coverage_consumer.py",
        "tests/native_go_coverage_contracts/run.py",
        "src/sc_neurocore/accel/go/internal/coverageprofile/snapshot.go",
    ):
        assert str(native_builder.repository / relative) in actual
    assert str(native_builder.go) in actual
    assert str(native_builder.go.parent / "gofmt") in actual


def test_real_oracle_class_from_another_source_refuses_before_model_construction(
    native_builder: NativeGo, tmp_path: Path
) -> None:
    """Use the actual imported class and a different real source copy, without running a model."""
    view = tmp_path / "other-source"
    copied = view / "src/sc_neurocore/neurons/models/mat.py"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(native_builder.repository / "src/sc_neurocore/neurons/models/mat.py", copied)
    output = native_builder.job / "artifacts/uncreated-oracle.json"
    driver = native_builder.job / "artifacts/oracle-origin-driver.py"
    driver.write_text(
        "import sys;from pathlib import Path;"
        "sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1])/'src')];"
        "from sc_neurocore.neurons.models.mat import MATNeuron;"
        "from tools.go_mat_coverage import write_oracle\n"
        "try:\n write_oracle(Path(sys.argv[2]),Path(sys.argv[3]))\n"
        "except RuntimeError as error:\n"
        " assert 'Actual Python oracle class came from another source' in str(error)\n"
        " print(str(error))\n"
        "else:\n raise RuntimeError('A different actual source was accepted')\n"
    )
    assert (
        native_builder.command(
            "real-oracle-origin",
            native_builder.python_script(
                driver, [str(native_builder.repository), str(view), str(output)]
            ),
            native_builder.job,
        )
        == 0
    )
    assert not output.exists()


def test_actual_os_signal_invokes_the_gate_cancellation_handler_and_restores_it(
    native_builder: NativeGo,
) -> None:
    """A real owned child receives SIGTERM and catches the actual gate refusal before exiting."""
    driver = native_builder.job / "artifacts/real-cancellation-driver.py"
    driver.write_text(
        "import os,signal,sys;sys.path.insert(0,sys.argv[1]);"
        "from tools.go_mat_coverage import _cancel;"
        "previous=signal.signal(signal.SIGTERM,_cancel)\n"
        "try:\n"
        " try:\n  os.kill(os.getpid(),signal.SIGTERM)\n"
        " except InterruptedError as error:\n"
        "  assert str(error)=='Native gate interrupted by SIGTERM';print(str(error))\n"
        " else:\n  raise RuntimeError('The actual cancellation signal was not handled')\n"
        "finally:\n signal.signal(signal.SIGTERM,previous)\n"
        "assert signal.getsignal(signal.SIGTERM)==previous\n"
    )
    assert (
        native_builder.command(
            "real-cancellation",
            native_builder.python_script(driver, [str(native_builder.repository)]),
            native_builder.job,
        )
        == 0
    )
