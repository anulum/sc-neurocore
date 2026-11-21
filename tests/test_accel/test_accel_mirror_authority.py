# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for acceleration mirror authority declarations

"""Hold the acceleration lanes' authority and mirror declarations to the files of the tree."""

from __future__ import annotations

from pathlib import Path

from sc_neurocore.accel.go import (
    BROAD_GO_SERVICE_NAMESPACE_GLOBS,
    MAINTAINED_GO_PYTHON_ENTRYPOINTS,
)
import sc_neurocore.accel.go as go_module
from sc_neurocore.accel.go.services import (
    GO_SERVICE_FILE_GLOBS,
    GO_SERVICE_PACKAGE_INIT_GLOB,
)
import sc_neurocore.accel.go.services as go_services_module
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
GO_ROOT = REPO_ROOT / "src" / "sc_neurocore" / "accel" / "go"
GO_SERVICES_ROOT = GO_ROOT / "services"
JULIA_ROOT = REPO_ROOT / "src" / "sc_neurocore" / "accel" / "julia"
MOJO_ROOT = REPO_ROOT / "src" / "sc_neurocore" / "accel" / "mojo"


def test_maintained_go_python_entrypoints_exist() -> None:
    """Every maintained Go entry point that Python loads is a file of the tree."""
    assert MAINTAINED_GO_PYTHON_ENTRYPOINTS
    for rel_path in MAINTAINED_GO_PYTHON_ENTRYPOINTS:
        assert (GO_ROOT / rel_path).exists(), rel_path


def test_broad_go_service_namespace_patterns_declared() -> None:
    """The broad Go service namespace declares its file and package patterns."""
    assert "services/*.go" in BROAD_GO_SERVICE_NAMESPACE_GLOBS
    assert "services/*/__init__.py" in BROAD_GO_SERVICE_NAMESPACE_GLOBS


def test_go_module_public_contract_shape() -> None:
    """The Go acceleration package exports its two declaration tuples."""
    assert "BROAD_GO_SERVICE_NAMESPACE_GLOBS" in go_module.__all__
    assert "MAINTAINED_GO_PYTHON_ENTRYPOINTS" in go_module.__all__


def test_go_service_namespace_patterns_cover_real_tree() -> None:
    """The Go service patterns are exactly these, and each names files of the tree."""
    assert GO_SERVICE_FILE_GLOBS == ("*.go", "*/*.go")
    assert GO_SERVICE_PACKAGE_INIT_GLOB == "*/__init__.py"
    for pattern in GO_SERVICE_FILE_GLOBS:
        assert list(GO_SERVICES_ROOT.glob(pattern)), pattern
    assert list(GO_SERVICES_ROOT.glob(GO_SERVICE_PACKAGE_INIT_GLOB))


def test_go_services_module_public_contract_shape() -> None:
    """The Go services package exports its two pattern declarations."""
    assert "GO_SERVICE_FILE_GLOBS" in go_services_module.__all__
    assert "GO_SERVICE_PACKAGE_INIT_GLOB" in go_services_module.__all__


def test_authoritative_julia_entrypoints_exist() -> None:
    """Every authoritative Julia entry point is a file of the tree."""
    assert AUTHORITATIVE_JULIA_ENTRYPOINTS
    for rel_path in AUTHORITATIVE_JULIA_ENTRYPOINTS:
        assert (JULIA_ROOT / rel_path).exists(), rel_path


def test_non_authoritative_julia_patterns_declared() -> None:
    """The Julia mirror zones named in the mirror-authority guide are declared."""
    assert "analysis/*.jl" in NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS
    assert "analysis_spike_stats/*.jl" in NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS
    assert "edge/*.jl" in NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS
    assert "studio/*.jl" in NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS
    assert "model_zoo/*.jl" in NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS
    assert "spike_codec/*.jl" in NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS


def test_authoritative_mojo_entrypoints_exist() -> None:
    """Every authoritative Mojo entry point is a file of the tree."""
    assert AUTHORITATIVE_MOJO_ENTRYPOINTS
    for rel_path in AUTHORITATIVE_MOJO_ENTRYPOINTS:
        assert (MOJO_ROOT / rel_path).exists(), rel_path


def test_non_authoritative_mojo_patterns_declared() -> None:
    """The Mojo kernels and the compiler transcript are declared as mirrors."""
    assert "kernels/*.mojo" in NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS
    assert "kernels/compiler.mojo" in NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS


def test_every_declared_mirror_pattern_names_files_of_the_tree() -> None:
    """A mirror declaration that names no file is stale, for example after its stub is deleted."""
    for root, patterns in (
        (JULIA_ROOT, NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS),
        (MOJO_ROOT, NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS),
    ):
        for pattern in patterns:
            assert list(root.glob(pattern)), (root.name, pattern)


def test_mojo_module_public_contract_shape() -> None:
    """The Mojo package exports its declarations and states why Mojo is absent."""
    assert isinstance(mojo_module._HAS_MOJO, bool)
    assert "AUTHORITATIVE_MOJO_ENTRYPOINTS" in mojo_module.__all__
    assert "_HAS_MOJO" in mojo_module.__all__
    if not mojo_module._HAS_MOJO:
        assert isinstance(mojo_module._mojo_import_reason, str)
        assert mojo_module._mojo_import_reason
