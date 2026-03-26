# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.cli

"""Tests for sc_neurocore.cli."""

import builtins
import types
from unittest import mock

from sc_neurocore.cli import _cmd_info, _cmd_studio, _format_engine_status, main


def _run_main(*argv: str) -> int:
    with mock.patch("sys.argv", ["sc-neurocore", *argv]):
        return main()


def _fake_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_version_flag(capsys):
    rc = _run_main("--version")
    assert rc == 0
    from sc_neurocore import __version__

    assert __version__ in capsys.readouterr().out


def test_info_command(capsys):
    fake_jax = _fake_module("jax", __version__="0.0-test")
    with mock.patch.dict("sys.modules", {"jax": fake_jax}):
        rc = _run_main("info")
    assert rc == 0
    out = capsys.readouterr().out
    assert "sc-neurocore" in out
    assert "Python" in out
    assert "NumPy" in out
    assert "JAX: 0.0-test" in out


def test_no_command_prints_help(capsys):
    rc = _run_main()
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_info_without_rust_engine(capsys):
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": None}):
        rc = _cmd_info()
    assert rc == 0
    assert "not available" in capsys.readouterr().out


def test_info_reports_engine_version_mismatch(capsys):
    fake = _fake_module(
        "sc_neurocore_engine",
        __version__="0.0.0",
        simd_tier=lambda: "mock-tier",
    )
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        rc = _cmd_info()
    assert rc == 0
    out = capsys.readouterr().out
    assert "version mismatch" in out
    assert "expected" in out


def test_info_ignores_broken_optional_jax_import(capsys):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax":
            raise AttributeError("broken jax")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        rc = _cmd_info()
    assert rc == 0
    assert "JAX:" not in capsys.readouterr().out


def test_info_ignores_broken_optional_numpy_import(capsys):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy":
            raise RuntimeError("broken numpy")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=fake_import):
        rc = _cmd_info()
    assert rc == 0
    assert "NumPy:" not in capsys.readouterr().out


def test_format_engine_status_without_simd_tier():
    fake = _fake_module("sc_neurocore_engine", __version__="3.13.0")
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        status = _format_engine_status("3.13.0")
    assert status == "Rust engine: 3.13.0 (unknown)"


def test_format_engine_status_with_broken_simd_tier():
    def explode():
        raise RuntimeError("no simd")

    fake = _fake_module(
        "sc_neurocore_engine",
        __version__="3.13.0",
        simd_tier=explode,
    )
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        status = _format_engine_status("3.13.0")
    assert status == "Rust engine: 3.13.0 (unknown)"


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


def test_studio_launches_uvicorn(capsys):
    with (
        mock.patch("uvicorn.run") as m_uvicorn,
        mock.patch("webbrowser.open") as m_browser,
    ):
        rc = _cmd_studio(port=8001)
    assert rc == 0
    m_uvicorn.assert_called_once()
    m_browser.assert_called_once_with("http://127.0.0.1:8001")


def test_studio_missing_fastapi(capsys):
    real_import = builtins.__import__

    def block_uvicorn(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "uvicorn":
            raise ImportError("No module named 'uvicorn'")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=block_uvicorn):
        rc = _cmd_studio(port=8001)
    assert rc == 1
    assert "pip install" in capsys.readouterr().out


def test_studio_command_via_main(capsys):
    with (
        mock.patch("sc_neurocore.cli._cmd_studio", return_value=0) as m_studio,
    ):
        rc = _run_main("studio")
    assert rc == 0
    m_studio.assert_called_once_with(8001)
