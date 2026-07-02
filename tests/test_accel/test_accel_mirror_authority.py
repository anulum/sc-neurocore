# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for acceleration mirror authority declarations

from __future__ import annotations

from pathlib import Path

from sc_neurocore.accel.julia import (
    AUTHORITATIVE_JULIA_ENTRYPOINTS,
    NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS,
)
from sc_neurocore.accel.mojo import (
    AUTHORITATIVE_MOJO_ENTRYPOINTS,
    NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS,
)
import sc_neurocore.accel.mojo as mojo_module


REPO_ROOT = Path(__file__).resolve().parents[2]
JULIA_ROOT = REPO_ROOT / "src" / "sc_neurocore" / "accel" / "julia"
MOJO_ROOT = REPO_ROOT / "src" / "sc_neurocore" / "accel" / "mojo"


def test_authoritative_julia_entrypoints_exist() -> None:
    assert AUTHORITATIVE_JULIA_ENTRYPOINTS
    for rel_path in AUTHORITATIVE_JULIA_ENTRYPOINTS:
        assert (JULIA_ROOT / rel_path).exists(), rel_path


def test_non_authoritative_julia_patterns_declared() -> None:
    assert "studio/*.jl" in NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS
    assert "model_zoo/*.jl" in NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS


def test_authoritative_mojo_entrypoints_exist() -> None:
    assert AUTHORITATIVE_MOJO_ENTRYPOINTS
    for rel_path in AUTHORITATIVE_MOJO_ENTRYPOINTS:
        assert (MOJO_ROOT / rel_path).exists(), rel_path


def test_non_authoritative_mojo_patterns_declared() -> None:
    assert "kernels/*.mojo" in NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS
    assert "kernels/app.mojo" in NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS


def test_mojo_module_public_contract_shape() -> None:
    assert isinstance(mojo_module._HAS_MOJO, bool)
    assert "AUTHORITATIVE_MOJO_ENTRYPOINTS" in mojo_module.__all__
    assert "_HAS_MOJO" in mojo_module.__all__
    if not mojo_module._HAS_MOJO:
        assert isinstance(mojo_module._mojo_import_reason, str)
        assert mojo_module._mojo_import_reason
