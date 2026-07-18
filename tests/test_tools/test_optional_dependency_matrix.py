# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optional dependency matrix contract tests

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI path.
    import tomli as tomllib  # type: ignore[no-redef]


class DependencyRow(NamedTuple):
    """Expected optional-dependency matrix row."""

    import_name: str
    distribution: str
    declared_extra: str | None
    test_paths: tuple[str, ...]


EXPECTED_ROWS = (
    DependencyRow(
        import_name="gdsfactory",
        distribution="gdsfactory>=9.0",
        declared_extra="optics",
        test_paths=(
            "tests/test_optics/test_gdsii.py",
            "tests/test_optics/test_photonic_emitter_branches.py",
            ".github/workflows/ci.yml",
        ),
    ),
    DependencyRow(
        import_name="neal",
        distribution="dwave-neal",
        declared_extra="annealing",
        test_paths=(
            "tests/test_bridges/test_quantum_annealing_neal_parity.py",
            ".github/workflows/ci.yml",
        ),
    ),
    DependencyRow(
        import_name="onnx",
        distribution="onnx",
        declared_extra="dev",
        test_paths=(
            "tests/test_export/test_onnx_exporter.py",
            "tests/test_export/test_onnx_export.py",
            ".github/workflows/ci.yml",
        ),
    ),
    DependencyRow(
        import_name="lava-nc",
        distribution="lava-nc",
        declared_extra="lava",
        test_paths=("tests/test_nir_neuromorphic_adapters.py",),
    ),
    DependencyRow(
        import_name="snntorch",
        distribution="snntorch",
        declared_extra=None,
        test_paths=(
            "tests/test_nir_bridge/test_scalar_broadcast.py",
            "docs/guides/nir_integration.md",
        ),
    ),
    DependencyRow(
        import_name="spikingjelly",
        distribution="spikingjelly",
        declared_extra=None,
        test_paths=(
            "tests/test_nir_bridge/test_spikingjelly_interop.py",
            "docs/guides/nir_integration.md",
        ),
    ),
    DependencyRow(
        import_name="cupy",
        distribution="cupy-cuda12x>=12.0",
        declared_extra="gpu",
        test_paths=("pyproject.toml", "docs/guides/faq.md", "docs/api/training.md"),
    ),
    DependencyRow(
        import_name="mpi4py",
        distribution="mpi4py>=3.0",
        declared_extra="mpi",
        test_paths=(
            "tests/test_mpi_runner_real.py",
            "tests/_mpi_helpers/mpi_runner_worker.py",
            ".github/workflows/ci.yml",
        ),
    ),
)


def _repo_root() -> Path:
    """Return the repository root."""

    return Path(__file__).resolve().parents[2]


def _optional_dependencies() -> dict[str, list[str]]:
    """Load optional dependency groups from project metadata."""

    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    return cast(dict[str, list[str]], pyproject["project"]["optional-dependencies"])


def _matrix_text() -> str:
    """Return the public optional-dependency matrix text."""

    return (_repo_root() / "docs" / "guides" / "optional_dependency_matrix.md").read_text(
        encoding="utf-8"
    )


def _normalized_requirements(extra: str) -> set[str]:
    """Return requirement strings without environment markers for one extra."""

    optional = _optional_dependencies()
    requirements = optional[extra]
    return {requirement.split(";", maxsplit=1)[0].strip() for requirement in requirements}


def test_optional_dependency_matrix_covers_audit_targets() -> None:
    """Ensure every audited optional dependency appears in docs with evidence."""

    text = _matrix_text()

    for row in EXPECTED_ROWS:
        assert row.import_name in text
        assert row.distribution in text
        for test_path in row.test_paths:
            assert test_path in text


def test_declared_optional_dependencies_match_pyproject() -> None:
    """Ensure documented declared extras match current project metadata."""

    text = _matrix_text()

    for row in EXPECTED_ROWS:
        if row.declared_extra is None:
            assert row.distribution in text
            assert "Not declared in `pyproject.toml`" in text
            continue
        assert row.declared_extra in _optional_dependencies()
        assert row.distribution in _normalized_requirements(row.declared_extra)


def test_matrix_is_linked_from_install_profiles_and_nav() -> None:
    """Keep the optional matrix reachable from public install docs and MkDocs."""

    install_profiles = (_repo_root() / "docs" / "guides" / "install_profiles.md").read_text(
        encoding="utf-8"
    )
    mkdocs = (_repo_root() / "mkdocs.yml").read_text(encoding="utf-8")

    assert "optional_dependency_matrix.md" in install_profiles
    assert "guides/optional_dependency_matrix.md" in mkdocs
