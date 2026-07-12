# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli dispatch tests

"""Exercise cli dispatch behaviour through the public CLI."""

from __future__ import annotations

import runpy
import sys

import pytest

from tests.cli_test_support import run_cli


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    rc = run_cli("--version")
    assert rc == 0
    from sc_neurocore import __version__

    assert __version__ in capsys.readouterr().out


def test_no_command_prints_grouped_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = run_cli()
    assert rc == 0
    output = capsys.readouterr().out
    assert "usage" in output.lower()
    assert "Model     info, compile, compile-nir, serve, map-nir" in output
    assert "Hardware  deploy, collect-synthesis, scnir, formal, hub-init" in output
    assert "Studio    studio and studio-* operator commands" in output


def test_command_help_is_progressively_disclosed(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        run_cli("compile-nir", "--help")
    output = capsys.readouterr().out
    assert "--data-width" in output
    assert "--fraction" in output
    assert "--identity-file" not in output


def test_module_entrypoint_dispatches_sys_argv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m sc_neurocore.cli`` exits with the public command status."""
    monkeypatch.setattr(sys, "argv", ["python -m sc_neurocore.cli", "--version"])

    with pytest.raises(SystemExit) as raised:
        runpy.run_module("sc_neurocore.cli", run_name="__main__")

    assert raised.value.code == 0
    assert "sc-neurocore" in capsys.readouterr().out
