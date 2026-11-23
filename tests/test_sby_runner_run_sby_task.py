# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRunSbyTask from former test_sby_runner.py

"""Focused suite: TestRunSbyTask from former test_sby_runner.py."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, call

import pytest
import subprocess
from pathlib import (
    Path,
)
from sc_neurocore.compiler import (
    _sby_runner,
)
from sc_neurocore.compiler._sby_runner import (
    run_sby_task,
)
from tests.sby_runner_support import (
    _FakeProc,
)


class TestRunSbyTask:
    """The subprocess boundary, exercised with a fake process."""

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Translate a process deadline into the public runtime error."""

        def _raise_timeout(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="sby", timeout=1.0)

        monkeypatch.setattr(_sby_runner, "_run_sby_process", _raise_timeout)
        with pytest.raises(RuntimeError, match="timed out after 1.0s"):
            run_sby_task(tmp_path, "x.sby", timeout_s=1.0)

    @pytest.mark.parametrize("stalls_on_shutdown", [False, True])
    def test_timeout_allows_symbiyosys_to_terminate_its_solvers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stalls_on_shutdown: bool
    ) -> None:
        """Signal the task before escalating so its solver groups can exit."""
        process = MagicMock(spec=subprocess.Popen)
        process.__enter__.return_value = process
        timeout = subprocess.TimeoutExpired(cmd="sby", timeout=1.0)
        shutdown = subprocess.TimeoutExpired(cmd="sby", timeout=3.0)
        process.communicate.side_effect = (timeout, shutdown if stalls_on_shutdown else ("", ""))
        monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)

        with pytest.raises(RuntimeError, match="timed out after 1.0s"):
            run_sby_task(tmp_path, "x.sby", timeout_s=1.0)

        process.assert_has_calls(
            [call.communicate(timeout=1.0), call.terminate(), call.communicate(timeout=3.0)]
        )
        assert process.kill.call_count == int(stalls_on_shutdown)

    def test_pass_run_has_no_counterexample(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Carry the successful proof report without a counterexample."""
        stdout = "SBY summary: engine_0\nDONE (PASS, rc=0)\n"
        monkeypatch.setattr(_sby_runner, "_run_sby_process", lambda *a, **k: _FakeProc(stdout, 0))
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
        """Preserve the solver's failed assertion and its trace location."""
        stdout = (
            "summary: counterexample trace: bad_bmc/engine_0/trace.vcd\n"
            "summary: failed assertion mon.sva_i at mon_sva.sv:3 step 6\n"
            "DONE (FAIL, rc=2)\n"
        )
        monkeypatch.setattr(_sby_runner, "_run_sby_process", lambda *a, **k: _FakeProc(stdout, 2))
        run = run_sby_task(tmp_path, "x.sby", timeout_s=5.0)
        assert run.verdict == "FAIL"
        assert run.counterexample is not None
        assert "failed assertion" in run.counterexample.lower()
        assert run.trace_path == str(tmp_path / "bad_bmc/engine_0/trace.vcd")

    def test_none_stdout_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Keep a missing subprocess report inconclusive."""
        monkeypatch.setattr(
            _sby_runner,
            "_run_sby_process",
            lambda *a, **k: _FakeProc(cast(Any, None), 1),
        )
        run = run_sby_task(tmp_path, "x.sby", timeout_s=5.0)
        assert run.verdict == "UNKNOWN"
        assert run.stdout == ""
