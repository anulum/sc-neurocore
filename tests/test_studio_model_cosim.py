# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio selected-model real RTL co-simulation tests

from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from typing import cast

import pytest

from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.studio.model_compile_configuration import (
    resolve_model_compile_configuration,
)
from sc_neurocore.studio.model_cosim import (
    STUDIO_COSIM_PARITY_SCHEMA_VERSION,
    _first_mismatch,
    _parse_trace,
    _resolve_tool,
    _run_checked,
    _tool_version,
    run_model_cosim,
)

HAS_COSIM_TOOLS = all(shutil.which(tool) is not None for tool in ("gcc", "iverilog", "vvp"))


def _map_configuration():
    return resolve_model_compile_configuration(
        {
            "model_name": "AdaptiveThresholdIFNeuron",
            "integrator": "map",
            "q_format": "Q8.8",
        }
    )


@pytest.mark.skipif(not HAS_COSIM_TOOLS, reason="GCC and Icarus Verilog are required")
def test_model_cosim_compares_every_real_rtl_state_cycle_bit_exactly() -> None:
    execution = run_model_cosim(_map_configuration(), current=10.0, n_steps=16)

    report = execution.report
    assert report["schema_version"] == STUDIO_COSIM_PARITY_SCHEMA_VERSION
    assert report["bit_exact"] is True
    assert report["first_mismatch"] is None
    assert report["sample_count"] == 16
    assert report["signals"] == ["spike_out", "v_out", "theta_out"]
    assert execution.rtl_trace == execution.reference_trace
    rtl = cast(dict[str, str], report["rtl"])
    reference = cast(dict[str, str], report["reference"])
    assert rtl["trace_sha256"] == reference["trace_sha256"]
    assert set(cast(dict[str, str], report["tools"])) == {"gcc", "iverilog", "vvp"}


def test_model_cosim_rejects_integrator_without_bit_true_reference() -> None:
    configuration = resolve_model_compile_configuration(
        {"model_name": "SCLapicqueLIFNeuron", "integrator": "exp_euler", "q_format": "Q8.8"}
    )

    with pytest.raises(ValueError, match="supports integrators euler, map"):
        run_model_cosim(configuration, current=10.0, n_steps=16)


def test_model_cosim_rejects_unavailable_external_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sc_neurocore.studio.model_cosim._resolve_tool",
        lambda name: None if name == "vvp" else f"/usr/bin/{name}",
    )

    with pytest.raises(RuntimeError, match="tools unavailable: vvp"):
        run_model_cosim(_map_configuration(), current=10.0, n_steps=16)


@pytest.mark.parametrize("current", [True, float("nan")])
def test_model_cosim_rejects_invalid_current(current: object) -> None:
    with pytest.raises(ValueError, match="current must be a finite number"):
        run_model_cosim(_map_configuration(), current=current, n_steps=16)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_steps", [True, 0, 2049])
def test_model_cosim_rejects_invalid_step_count(n_steps: object) -> None:
    with pytest.raises(ValueError, match="n_steps must be between"):
        run_model_cosim(_map_configuration(), current=1.0, n_steps=n_steps)  # type: ignore[arg-type]


@pytest.mark.parametrize("q_format", [QFormat(7, 8), QFormat(32, 32)])
def test_model_cosim_rejects_unsupported_q_width(q_format: QFormat) -> None:
    configuration = replace(_map_configuration(), q_format=q_format)

    with pytest.raises(ValueError, match="8, 16 or 32-bit"):
        run_model_cosim(configuration, current=1.0, n_steps=4)


def test_model_cosim_tool_and_subprocess_failures_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Unsupported Studio co-simulation tool"):
        _resolve_tool("shell")

    monkeypatch.setattr(
        "sc_neurocore.studio.model_cosim.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["gcc"], returncode=2, stdout="", stderr="compile\nfailed"
        ),
    )
    with pytest.raises(RuntimeError, match="exited 2: compile failed"):
        _run_checked(["/usr/bin/gcc"], timeout_seconds=1)

    monkeypatch.setattr(
        "sc_neurocore.studio.model_cosim.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    with pytest.raises(RuntimeError, match="command failed: gcc"):
        _run_checked(["/usr/bin/gcc"], timeout_seconds=1)

    assert _tool_version("/usr/bin/gcc", "gcc") == "available-version-unreported"


def test_model_cosim_rejects_incomplete_external_trace() -> None:
    with pytest.raises(RuntimeError, match="emitted 1 of 2 trace rows"):
        _parse_trace("0 1\ndiagnostic", n_steps=2, n_signals=2, label="RTL")


def test_model_cosim_reports_first_cycle_and_signal_mismatch() -> None:
    mismatch = _first_mismatch(
        [[0, 1, 2], [1, 4, 6]],
        [[0, 1, 2], [0, 4, 5]],
        ["spike_out", "v_out", "theta_out"],
    )

    assert mismatch == {
        "cycle": 2,
        "reference": {"spike_out": 0, "v_out": 4, "theta_out": 5},
        "rtl": {"spike_out": 1, "v_out": 4, "theta_out": 6},
        "signals": ["spike_out", "theta_out"],
    }


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"model_name": "SCLapicqueLIFNeuron", "module_name": "bad module"}, "Verilog identifier"),
        ({"model_name": "SCLapicqueLIFNeuron", "q_format": "Q1.0"}, "between 2 and 64"),
        ({"model_name": "SCLapicqueLIFNeuron", "params": {"missing": 1.0}}, "Unknown schema"),
    ],
)
def test_shared_model_compile_configuration_fails_closed(
    payload: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_model_compile_configuration(payload)
