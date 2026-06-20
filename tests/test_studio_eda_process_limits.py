# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio EDA process-limit contracts

"""Contract tests for Studio EDA child-process resource ceilings."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from sc_neurocore.studio import synthesis
from sc_neurocore.studio.synthesis import EdaProcessLimits, run_pnr, run_synthesis


def test_eda_process_limits_reject_non_positive_values() -> None:
    """Configured EDA process ceilings must be positive when present."""

    with pytest.raises(ValueError, match="CPU limit"):
        EdaProcessLimits(cpu_seconds=0)
    with pytest.raises(ValueError, match="memory limit"):
        EdaProcessLimits(address_space_bytes=0)


def test_run_synthesis_passes_posix_preexec_when_limits_are_configured(
    monkeypatch: MonkeyPatch,
) -> None:
    """Yosys invocation receives a POSIX pre-exec limiter on supported hosts."""

    captured_preexec: list[Any] = []

    def resolve_tool(name: str) -> str | None:
        assert name == "yosys"
        return "/bin/true"

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        shell: bool,
        text: bool,
        timeout: float,
        preexec_fn: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert shell is False
        assert text is True
        assert timeout == 60
        captured_preexec.append(preexec_fn)
        script_path = Path(args[2])
        (script_path.parent / "design.json").write_text('{"modules": {}}', encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(synthesis, "_resolve_eda_tool", resolve_tool)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_synthesis(
        "module test(); endmodule",
        "ice40",
        process_limits=EdaProcessLimits(cpu_seconds=3.0, address_space_bytes=64 * 1024 * 1024),
    )

    assert result["success"] is True
    assert len(captured_preexec) == 1
    if os.name == "posix":
        assert callable(captured_preexec[0])
    else:
        assert captured_preexec[0] is None


def test_run_pnr_passes_posix_preexec_when_limits_are_configured(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """nextpnr invocation receives the configured child-process limiter."""

    captured_preexec: list[Any] = []
    netlist_path = tmp_path / "design.json"
    netlist_path.write_text("{}", encoding="utf-8")

    def resolve_tool(name: str) -> str | None:
        assert name == "nextpnr-ice40"
        return "/bin/true"

    def fake_run(
        args: list[str],
        *,
        capture_output: bool,
        shell: bool,
        text: bool,
        timeout: float,
        preexec_fn: Any | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert "--json" in args
        assert capture_output is True
        assert shell is False
        assert text is True
        assert timeout == 120
        captured_preexec.append(preexec_fn)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="Max frequency: 12.5 MHz\n",
            stderr="",
        )

    monkeypatch.setattr(synthesis, "_resolve_eda_tool", resolve_tool)
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_pnr(
        str(netlist_path),
        "ice40",
        process_limits=EdaProcessLimits(cpu_seconds=5.0, address_space_bytes=64 * 1024 * 1024),
    )

    assert result["success"] is True
    assert result["max_freq_mhz"] == 12.5
    assert len(captured_preexec) == 1
    if os.name == "posix":
        assert callable(captured_preexec[0])
    else:
        assert captured_preexec[0] is None
