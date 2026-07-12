# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli info tests

"""Exercise cli info behaviour through the public CLI."""

from __future__ import annotations

import importlib.metadata
from unittest import mock

import pytest

from tests.cli_test_support import fake_module, run_cli


def test_info_command_reports_loaded_optional_dependency(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_jax = fake_module("jax", __version__="0.0-test")
    with mock.patch.dict("sys.modules", {"jax": fake_jax}):
        rc = run_cli("info")
    assert rc == 0
    out = capsys.readouterr().out
    assert "sc-neurocore" in out
    assert "Python" in out
    assert "NumPy" in out
    assert "JAX: 0.0-test" in out


def test_info_without_rust_engine(capsys: pytest.CaptureFixture[str]) -> None:
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": None}):
        rc = run_cli("info")
    assert rc == 0
    assert "not available" in capsys.readouterr().out


def test_info_reports_engine_version_mismatch(capsys: pytest.CaptureFixture[str]) -> None:
    fake = fake_module(
        "sc_neurocore_engine",
        __version__="0.0.0",
        simd_tier=lambda: "mock-tier",
    )
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        rc = run_cli("info")
    assert rc == 0
    out = capsys.readouterr().out
    assert "version mismatch" in out
    assert "expected" in out


def test_info_uses_metadata_without_importing_optional_jax(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_version(name: str) -> str:
        if name == "jax":
            return "0.0-meta"
        if name == "numpy":
            return "0.0-numpy"
        raise importlib.metadata.PackageNotFoundError(name)

    with (
        mock.patch.dict("sys.modules", {"jax": None}),
        mock.patch(
            "sc_neurocore.cli.commands.info.importlib.metadata.version",
            side_effect=fake_version,
        ),
    ):
        rc = run_cli("info")
    assert rc == 0
    out = capsys.readouterr().out
    assert "JAX: 0.0-meta" in out


def test_info_ignores_missing_optional_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        mock.patch.dict("sys.modules", {"numpy": None, "jax": None}),
        mock.patch(
            "sc_neurocore.cli.commands.info.importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("missing"),
        ),
    ):
        rc = run_cli("info")
    assert rc == 0
    assert "NumPy:" not in capsys.readouterr().out


def test_info_reports_unknown_simd_tier(capsys: pytest.CaptureFixture[str]) -> None:
    from sc_neurocore import __version__

    fake = fake_module("sc_neurocore_engine", __version__=__version__)
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        assert run_cli("info") == 0
    assert f"Rust engine: {__version__} (unknown)" in capsys.readouterr().out


def test_info_contains_broken_simd_probe(capsys: pytest.CaptureFixture[str]) -> None:
    from sc_neurocore import __version__

    def explode() -> str:
        raise RuntimeError("no simd")

    fake = fake_module(
        "sc_neurocore_engine",
        __version__=__version__,
        simd_tier=explode,
    )
    with mock.patch.dict("sys.modules", {"sc_neurocore_engine": fake}):
        assert run_cli("info") == 0
    assert f"Rust engine: {__version__} (unknown)" in capsys.readouterr().out


def test_info_ignores_loaded_dependency_without_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A loaded namespace package without version metadata is omitted safely."""
    fake_numpy = fake_module("numpy")
    with mock.patch.dict("sys.modules", {"numpy": fake_numpy}):
        assert run_cli("info") == 0
    assert "NumPy:" not in capsys.readouterr().out
