# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The physical-GPU evidence record behind the coverage platform exception

"""Hold the evidence recorder to the three reports of a physical-GPU run and its refusals.

Each test writes a checkout with a register and the sources it names, plus the adapter JSON,
the JUnit report and the ``coverage json`` report that a physical-GPU run leaves, then runs the
recorder's command line. One test checks the record this repository carries.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tools.coverage_exception_gate as gate
import tools.gpu_waveform_encoder_evidence as evidence

#: The JUnit report of a passing run of the GPU test, after a same-named test of another module.
PASSED = (
    '<testsuites><testsuite name="pytest"><testcase classname="tests.other" '
    'name="test_installed_gpu_encoder_uses_the_reported_physical_backend"><failure/></testcase>'
    "<testcase "
    'classname="tests.test_waveform_stream" '
    'name="test_installed_gpu_encoder_uses_the_reported_physical_backend" time="0.4"/>'
    "</testsuite></testsuites>"
)


def _checkout(root: Path) -> None:
    """Write a checkout with a register for one excepted line and its two sources."""
    (root / "src/pkg").mkdir(parents=True)
    (root / "tools").mkdir()
    (root / "src/pkg/_gpu.py").write_text('"""GPU."""\n\n\n    return coder\n', encoding="utf-8")
    (root / "engine.rs").write_text("fn gpu() {}\n", encoding="utf-8")
    register = {
        "schema": gate.REGISTER_SCHEMA,
        "decisions": ["523df20b", "fd2fcd90"],
        "evidence": "tools/evidence.json",
        "source_set": ["engine.rs", "src/pkg/_gpu.py"],
        "entries": [{"path": "src/pkg/_gpu.py", "line": 4, "source_sha256": "0" * 64}],
    }
    (root / gate.REGISTER).write_text(json.dumps(register), encoding="utf-8")


def _inputs(
    root: Path,
    *,
    adapter: dict[str, Any] | None = None,
    junit: str = PASSED,
    executed: list[int] | None = None,
) -> list[str]:
    """Write the three reports of a run and return the recorder's command line for them.

    Parameters
    ----------
    root : Path
        The checkout.
    adapter : dict or None
        The adapter JSON; a physical Vulkan adapter when None.
    junit : str
        The JUnit report.
    executed : list of int or None
        The executed lines of the excepted file; the registered line when None.

    Returns
    -------
    list of str
        The arguments for :func:`tools.gpu_waveform_encoder_evidence.main`.
    """
    adapter = adapter or {"name": "Physical GPU", "api": "Vulkan", "engine_version": "4.2.0"}
    files = {"src/pkg/_gpu.py": {"executed_lines": [4] if executed is None else executed}}
    (root / "adapter.json").write_text(json.dumps(adapter), encoding="utf-8")
    (root / "junit.xml").write_text(junit, encoding="utf-8")
    (root / "coverage.json").write_text(
        json.dumps({"meta": {"version": "7.17.3"}, "files": files}), encoding="utf-8"
    )
    return [
        "--adapter",
        str(root / "adapter.json"),
        "--junit",
        str(root / "junit.xml"),
        "--coverage-json",
        str(root / "coverage.json"),
        "--python-version",
        "3.12.16",
        "--root",
        str(root),
    ]


def test_a_passing_physical_gpu_run_is_recorded_with_its_source_digests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The record names the adapter, the passing test, the covered line and each source's digest.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the confirmation.
    """
    _checkout(tmp_path)
    assert evidence.main(_inputs(tmp_path)) == 0
    assert "recorded Physical GPU" in capsys.readouterr().out
    record = json.loads((tmp_path / "tools/evidence.json").read_text(encoding="utf-8"))
    assert record["schema"] == gate.EVIDENCE_SCHEMA
    assert record["decisions"] == ["523df20b", "fd2fcd90"]
    assert record["adapter"] == {"name": "Physical GPU", "api": "Vulkan"}
    assert record["test"] == {"nodeid": evidence.GPU_TEST, "outcome": "passed"}
    assert record["excepted_lines_covered"] == [{"path": "src/pkg/_gpu.py", "line": 4}]
    assert record["source_sha256"]["engine.rs"] == hashlib.sha256(b"fn gpu() {}\n").hexdigest()
    assert record["tool_versions"] == {
        "python": "3.12.16",
        "sc_neurocore_engine": "4.2.0",
        "coverage": "7.17.3",
    }
    assert record["recorded_at"].endswith("Z")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"adapter": {"name": "llvmpipe", "api": "Gl"}},
            "is not a physical adapter's graphics API",
        ),
        ({"adapter": {"name": "", "api": "Vulkan"}}, "is not a physical adapter's graphics API"),
        (
            {"adapter": {"name": "llvmpipe (LLVM 20.1.2, 256 bits)", "api": "Vulkan"}},
            "is a known software adapter (llvmpipe)",
        ),
        (
            {"adapter": {"name": "Google SwiftShader Device", "api": "Vulkan"}},
            "is a known software adapter (swiftshader)",
        ),
        (
            {"adapter": {"name": "Microsoft Basic Render Driver", "api": "Dx12"}},
            "is a known software adapter (microsoft basic render)",
        ),
        (
            {"junit": PASSED.replace('time="0.4"/>', 'time="0.4"><skipped/></testcase>')},
            "skipped; the evidence needs it",
        ),
        (
            {"junit": PASSED.replace('time="0.4"/>', 'time="0.4"><failure/></testcase>')},
            "failed; the evidence needs it",
        ),
        (
            {"junit": PASSED.replace('time="0.4"/>', 'time="0.4"><error/></testcase>')},
            "failed; the evidence needs it",
        ),
        ({"junit": "<testsuites/>"}, "has no result for"),
        ({"junit": "<testsuites"}, "cannot be parsed"),
        ({"executed": [1, 2]}, "src/pkg/_gpu.py:4 did not run on the GPU"),
    ],
)
def test_a_run_that_does_not_prove_the_line_on_a_physical_gpu_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], overrides: dict[str, Any], message: str
) -> None:
    """No record is written unless a physical adapter passed the test and ran the line.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the refusal.
    overrides : dict
        The reports to replace.
    message : str
        The refusal the recorder must print.
    """
    _checkout(tmp_path)
    assert evidence.main(_inputs(tmp_path, **overrides)) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "tools/evidence.json").exists()


def test_a_missing_gpu_source_or_register_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A source the register names must exist, and the register itself must be readable.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    capsys : pytest.CaptureFixture[str]
        Captures the refusals.
    """
    _checkout(tmp_path)
    arguments = _inputs(tmp_path)
    (tmp_path / "engine.rs").unlink()
    assert evidence.main(arguments) == 2
    assert "engine.rs of the GPU sources cannot be read" in capsys.readouterr().err
    (tmp_path / gate.REGISTER).write_text('{"schema": "other.v1"}', encoding="utf-8")
    assert evidence.main(arguments) == 2
    assert "register declares 'other.v1'" in capsys.readouterr().err
    with pytest.raises(evidence.EvidenceError, match="register declares"):
        evidence.build_record(tmp_path, {}, PASSED, {}, "3.12.16", "2026-11-22T00:00:00Z")


def test_this_repository_record_is_a_physical_gpu_run_of_the_registered_line() -> None:
    """The committed record ran the GPU test on a physical adapter and covered the excepted line."""
    register, entries = gate.load_register(gate.ROOT)
    record = json.loads((gate.ROOT / register["evidence"]).read_text(encoding="utf-8"))
    assert record["test"]["outcome"] == "passed"
    assert record["adapter"]["api"] in evidence.PHYSICAL_APIS
    assert record["excepted_lines_covered"] == [
        {"path": entry.path, "line": entry.line} for entry in entries
    ]
    assert sorted(record["source_sha256"]) == sorted(register["source_set"])


def test_the_recorder_runs_as_a_script(tmp_path: Path) -> None:
    """``python tools/gpu_waveform_encoder_evidence.py`` exits 2 for a refused run.

    Parameters
    ----------
    tmp_path : Path
        The checkout.
    """
    _checkout(tmp_path)
    arguments = _inputs(tmp_path, adapter={"name": "llvmpipe", "api": "Gl"})
    done = subprocess.run(
        [sys.executable, str(gate.ROOT / "tools/gpu_waveform_encoder_evidence.py"), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=gate.ROOT,
    )
    assert done.returncode == 2
    assert "is not a physical adapter's graphics API" in done.stderr
