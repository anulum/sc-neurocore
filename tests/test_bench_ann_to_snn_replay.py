# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete dense IF comparison command acceptance

"""Exercise the real five-runtime command, actual provider refusal and argument admission."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tests.test_accel_mojo_if_abi import library as mojo_library
from tests.test_conversion_go_native import library as go_library
from tests.test_conversion_native_replay import library as rust_library

__all__ = ["ROOT", "SCRIPT", "environment", "rust_library", "go_library", "mojo_library"]

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks/bench_ann_to_snn_replay.py"


@pytest.fixture(scope="module")
def environment(
    rust_library: Path,
    go_library: Path,
    mojo_library: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, str]:
    """Configure actual compiled providers and an owned matching offline Julia project."""
    executables = sorted((Path.home() / ".julia/juliaup").glob("julia-1.11.*/bin/julia"))
    assert executables, "installed Julia required"
    project = tmp_path_factory.mktemp("comparison-julia")
    version = importlib.metadata.version("juliacall")
    (project / "Project.toml").write_text(
        '[deps]\nPythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"\n'
        f'[compat]\nPythonCall = "={version}"\n'
    )
    configured = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (str(ROOT / "src"), str(ROOT / "benchmarks"), os.environ.get("PYTHONPATH", ""))
        ),
        SC_NEUROCORE_IF_RUST_LIB=str(rust_library),
        SC_NEUROCORE_IF_GO_LIB=str(go_library),
        SC_NEUROCORE_IF_MOJO_LIB=str(mojo_library),
        SC_NEUROCORE_IF_JULIA_ENABLED="1",
        PYTHON_JULIACALL_EXE=str(executables[-1]),
        PYTHON_JULIACALL_PROJECT=str(project),
        PYTHON_JULIACALL_THREADS="1",
        PYTHON_JULIACALL_HANDLE_SIGNALS="yes",
        JULIA_CONDAPKG_BACKEND="Null",
        JULIA_PKG_OFFLINE="true",
    )
    subprocess.run(
        [
            str(executables[-1]),
            f"--project={project}",
            "--startup-file=no",
            "-e",
            "using Pkg; Pkg.offline(true); Pkg.resolve()",
        ],
        env=configured,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return configured


def test_complete_public_command_binds_all_five_runtime_receipts(
    environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Publish all twenty complete per-provider records through the programmatic CLI entry."""
    output = tmp_path / "comparison.json"
    cpu = min(os.sched_getaffinity(0))
    program = (
        "import sys; from bench_ann_to_snn_replay import main; raise SystemExit(main(sys.argv[1:]))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            "--output",
            str(output),
            "--cpu",
            str(cpu),
            "--samples",
            "3",
            "--warmup",
            "1",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert record["kernel"] == "dense-if-f64-sequential-v1"
    assert record["meta"]["cpu_affinity"] == [cpu]
    providers = record["backends"]
    assert set(providers) == {"numpy", "rust", "go", "mojo", "julia"}
    baseline = providers["numpy"]["cases"]
    assert len(baseline) == 20
    for backend, entry in providers.items():
        assert entry["full_bit_parity"] and entry["median_call_ms"] > 0
        assert len(entry["cases"]) == 20
        for case, expected in zip(entry["cases"], baseline, strict=True):
            assert (case["name"], case["input_sha256"], case["response_sha256"]) == (
                expected["name"],
                expected["input_sha256"],
                expected["response_sha256"],
            )
            assert len(case["samples_ns"]) == 3 and min(case["samples_ns"]) > 0
            assert case["first_call_ns"] > 0
        assert output.with_suffix(f".{backend}.stdout").is_file()
        assert output.with_suffix(f".{backend}.stderr").is_file()
    for path, digest in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for backend in ("RUST", "GO", "MOJO"):
        variable = f"SC_NEUROCORE_IF_{backend}_LIB"
        assert (
            record["artifact_sha256"][variable]
            == hashlib.sha256(Path(environment[variable]).read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("argument,value", [("--samples", "2"), ("--warmup", "0"), ("--cpu", "-1")])
def test_invalid_capture_arguments_preserve_existing_report(
    tmp_path: Path,
    argument: str,
    value: str,
) -> None:
    """Refuse incomplete measurement budgets or disallowed affinity before publication."""
    output = tmp_path / "report.json"
    output.write_text("prior owner report")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--cpu",
            str(min(os.sched_getaffinity(0))),
            argument,
            value,
        ],
        cwd=ROOT,
        env=dict(
            os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
        ),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert output.read_text() == "prior owner report"


def test_missing_installed_provider_prevents_comparison_publication(
    environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Refuse an absent configured library before running or publishing an incomplete comparison."""
    settings = dict(environment)
    settings["SC_NEUROCORE_IF_RUST_LIB"] = str(tmp_path / "absent.so")
    output = tmp_path / "report.json"
    output.write_text("prior owner report")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--cpu",
            str(min(os.sched_getaffinity(0))),
        ],
        cwd=ROOT,
        env=settings,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and "absent.so" in result.stderr
    assert output.read_text() == "prior owner report"
