# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRunSbyTask from former test_sby_runner.py

"""Focused suite: TestRunSbyTask from former test_sby_runner.py."""

from __future__ import annotations

from tests.sby_runner_support import *  # noqa: F403


class TestRunSbyTask:
    """The subprocess boundary, exercised with a fake process."""

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def _raise_timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="sby", timeout=1.0)

        monkeypatch.setattr(_sby_runner.subprocess, "run", _raise_timeout)
        with pytest.raises(RuntimeError, match="timed out after 1.0s"):
            run_sby_task(tmp_path, "x.sby", timeout_s=1.0)

    def test_pass_run_has_no_counterexample(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stdout = "SBY summary: engine_0\nDONE (PASS, rc=0)\n"
        monkeypatch.setattr(_sby_runner.subprocess, "run", lambda *a, **k: _FakeProc(stdout, 0))
        run = run_sby_task(tmp_path, "x.sby", timeout_s=5.0)
        assert run.verdict == "PASS"
        assert run.rc == 0
        assert run.returncode == 0
        assert run.counterexample is None
        assert run.trace_path is None
        assert run.summary == ["SBY summary: engine_0"]

    def test_fail_run_extracts_counterexample_and_trace(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        stdout = (
            "summary: counterexample trace: bad_bmc/engine_0/trace.vcd\n"
            "summary: failed assertion mon.sva_i at mon_sva.sv:3 step 6\n"
            "DONE (FAIL, rc=2)\n"
        )
        monkeypatch.setattr(_sby_runner.subprocess, "run", lambda *a, **k: _FakeProc(stdout, 2))
        run = run_sby_task(tmp_path, "x.sby", timeout_s=5.0)
        assert run.verdict == "FAIL"
        assert run.counterexample is not None
        assert "failed assertion" in run.counterexample.lower()
        assert run.trace_path == str(tmp_path / "bad_bmc/engine_0/trace.vcd")

    def test_none_stdout_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            _sby_runner.subprocess,
            "run",
            lambda *a, **k: _FakeProc(None, 1),  # type: ignore[arg-type]
        )
        run = run_sby_task(tmp_path, "x.sby", timeout_s=5.0)
        assert run.verdict == "UNKNOWN"
        assert run.stdout == ""
