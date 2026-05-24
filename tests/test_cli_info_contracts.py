# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for CLI info contracts

"""Contracts for CLI info reporting with optional runtime modules."""

from __future__ import annotations

import sys
import types


def test_cli_info_reports_engine_when_module_is_available() -> None:
    from sc_neurocore.cli import _cmd_info

    fake = types.ModuleType("sc_neurocore_engine")
    fake.__version__ = "0.0.0-test"
    fake.simd_tier = lambda: "test"
    sys.modules["sc_neurocore_engine"] = fake
    try:
        assert _cmd_info() == 0
    finally:
        del sys.modules["sc_neurocore_engine"]


def test_cli_info_reports_jax_when_module_is_available(capsys) -> None:
    from sc_neurocore.cli import _cmd_info

    fake_jax = types.ModuleType("jax")
    fake_jax.__version__ = "0.0.0-test"
    sys.modules["jax"] = fake_jax
    try:
        assert _cmd_info() == 0
        assert "JAX" in capsys.readouterr().out
    finally:
        del sys.modules["jax"]
