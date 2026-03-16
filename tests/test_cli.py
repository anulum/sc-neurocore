# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Tests for sc_neurocore.cli."""

from unittest import mock

from sc_neurocore.cli import main, _cmd_info


def _run_main(*argv: str) -> int:
    with mock.patch("sys.argv", ["sc-neurocore", *argv]):
        return main()


def test_version_flag(capsys):
    rc = _run_main("--version")
    assert rc == 0
    from sc_neurocore import __version__

    assert __version__ in capsys.readouterr().out


def test_info_command(capsys):
    rc = _run_main("info")
    assert rc == 0
    out = capsys.readouterr().out
    assert "sc-neurocore" in out
    assert "Python" in out
    assert "NumPy" in out


def test_no_command_prints_help(capsys):
    rc = _run_main()
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_info_without_rust_engine(capsys):
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": None}):
        rc = _cmd_info()
    assert rc == 0
    assert "not available" in capsys.readouterr().out


def test_benchmark_delegates_to_subprocess():
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(returncode=0)
        rc = _run_main("benchmark")
    assert rc == 0
    m.assert_called_once()


def test_preflight_delegates_to_subprocess():
    with mock.patch("subprocess.run") as m:
        m.return_value = mock.Mock(returncode=0)
        rc = _run_main("preflight")
    assert rc == 0
    m.assert_called_once()
