# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — generated-RTL spike execution failure contracts

"""Exercise generated-neuron RTL failure handling through real OS processes."""

from __future__ import annotations

import os
import shlex
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.cosim_reference_fitzhugh_nagumo import _fitzhugh_nagumo_substep_neuron
from tests.cosim_rtl_spike_execution import (
    _neuron_verilog_spike_count_q1616,
    _verilog_spike_count_generic,
    _verilog_spike_count_q1616,
    _verilog_spike_count_q412,
)

_SpikeRunner = Callable[[], int]


def _q412_lif_spike_count() -> int:
    return _verilog_spike_count_q412("lif", n_steps=1, current=0.0)


def _q1616_lif_spike_count() -> int:
    return _verilog_spike_count_q1616("lif", n_steps=1, current=0.0)


def _q1616_equation_neuron_spike_count() -> int:
    return _neuron_verilog_spike_count_q1616(
        _fitzhugh_nagumo_substep_neuron(1),
        n_steps=1,
        current=0.0,
        module_name="sc_fitzhugh_nagumo_failure_contract",
    )


def _generic_lif_spike_count() -> int:
    return _verilog_spike_count_generic(
        "lif",
        n_steps=1,
        current=0.0,
        data_width=16,
        fraction=8,
    )


_SPIKE_RUNNERS: tuple[tuple[str, _SpikeRunner], ...] = (
    ("q412-schema", _q412_lif_spike_count),
    ("q1616-schema", _q1616_lif_spike_count),
    ("q1616-equation-neuron", _q1616_equation_neuron_spike_count),
    ("generic-precision", _generic_lif_spike_count),
)


def _required_tool(name: str) -> Path:
    tool = shutil.which(name)
    if tool is None:
        pytest.fail(f"required co-simulation tool is unavailable: {name}")
    return Path(tool)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _install_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    body: str,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    _write_executable(bin_dir / name, body)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.mark.parametrize(
    "runner",
    [runner for _, runner in _SPIKE_RUNNERS],
    ids=[name for name, _ in _SPIKE_RUNNERS],
)
def test_compile_failure_reports_real_iverilog_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: _SpikeRunner,
) -> None:
    iverilog = shlex.quote(str(_required_tool("iverilog")))
    _install_proxy(
        tmp_path,
        monkeypatch,
        name="iverilog",
        body=f'exec {iverilog} -sc-neurocore-invalid-option "$@"',
    )

    with pytest.raises(RuntimeError, match="iverilog compile failed"):
        runner()


@pytest.mark.parametrize(
    "runner",
    [runner for _, runner in _SPIKE_RUNNERS],
    ids=[name for name, _ in _SPIKE_RUNNERS],
)
def test_simulation_failure_reports_real_vvp_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: _SpikeRunner,
) -> None:
    vvp = shlex.quote(str(_required_tool("vvp")))
    missing_image = shlex.quote(str(tmp_path / "missing-simulation-image"))
    _install_proxy(
        tmp_path,
        monkeypatch,
        name="vvp",
        body=f"exec {vvp} {missing_image}",
    )

    with pytest.raises(RuntimeError, match="vvp simulation failed"):
        runner()


@pytest.mark.parametrize(
    "runner",
    [runner for _, runner in _SPIKE_RUNNERS],
    ids=[name for name, _ in _SPIKE_RUNNERS],
)
def test_malformed_real_simulation_output_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: _SpikeRunner,
) -> None:
    vvp = shlex.quote(str(_required_tool("vvp")))
    _install_proxy(
        tmp_path,
        monkeypatch,
        name="vvp",
        body=(
            f'{vvp} "$@" >/dev/null\n'
            "printf '%s\\n' 'simulation completed without a spike summary'"
        ),
    )

    with pytest.raises(RuntimeError, match="Could not parse spike count"):
        runner()
