# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (probes_and_metadata) from former test_bench_iqif.py

from __future__ import annotations

from tests.bench_iqif_support import *  # noqa: F403

def test_backend_probes_report_each_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind every backend name to its real availability probe."""
    monkeypatch.setattr(backends, "_HAS_RUST", True)
    monkeypatch.setattr(backends, "ensure_julia_loaded", lambda: False)
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: True)
    monkeypatch.setattr(backends, "ensure_mojo_loaded", lambda: False)

    assert benchmark._probe_backend("python") == (True, "")
    assert benchmark._probe_backend("rust") == (True, "")
    assert benchmark._probe_backend("julia")[0] is False
    assert benchmark._probe_backend("go") == (True, "")
    assert benchmark._probe_backend("mojo")[0] is False


def test_host_metadata_helpers_use_portable_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Linux metadata and PATH entries remain explicit in evidence."""
    original_read_text = Path.read_text

    def selective_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == Path("/proc/cpuinfo") or path == tmp_path / "missing":
            raise OSError("metadata unavailable")
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", selective_read)
    monkeypatch.setattr(platform, "processor", lambda: "")
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    fallback = tmp_path / "runtime"
    fallback.touch()
    assert benchmark._cpu_model() == "unknown"
    assert benchmark._read_optional(tmp_path / "missing") == "unavailable"
    assert benchmark._tool_path("runtime", fallback) == str(fallback)
    assert benchmark._tool_path("runtime", tmp_path / "absent") is None


def test_tool_version_reports_empty_and_failed_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime provenance distinguishes absence and execution failure."""
    assert benchmark._tool_version([]) == "unavailable"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot execute")),
    )
    assert benchmark._tool_version(["missing"]) == "unavailable"

    completed = subprocess.CompletedProcess(["runtime"], 7, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed)
    assert benchmark._tool_version(["runtime"]) == "exit 7"


def test_rust_safety_gate_reports_compile_and_execution_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark cannot promote when its standalone safety proof fails."""
    failure = subprocess.CompletedProcess(["rustc"], 1, stdout="", stderr="compile failed")
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: failure)
    result = benchmark._verify_rust_safety()
    assert result["passed"] is False
    assert result["returncode"] == 1
    assert result["output_tail"] == ["compile failed"]

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("rustc unavailable")),
    )
    result = benchmark._verify_rust_safety()
    assert result["passed"] is False
    assert result["returncode"] == -1
    assert result["output_tail"] == ["rustc unavailable"]


def test_allowed_unavailable_backend_is_recorded_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit diagnostic override retains an unavailable backend row."""
    monkeypatch.setattr(benchmark, "BACKENDS", ("python", "mojo"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(
        benchmark,
        "_probe_backend",
        lambda backend: (backend == "python", "missing" if backend == "mojo" else ""),
    )
    monkeypatch.setattr(benchmark, "_measure_backend", _measured_python)
    monkeypatch.setattr(benchmark, "_environment", lambda _load: {})
    monkeypatch.setattr(benchmark, "_source_hashes", lambda: {})
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)
    output = tmp_path / "partial.json"

    assert benchmark.main(["--json", str(output), "--allow-unavailable-backends"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backends"]["mojo"] == {
        "available": False,
        "used": False,
        "unavailable_reason": "missing",
    }
