# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — pip requirements files mirror the package metadata

"""Keep the pip requirements files that the README offers in step with pyproject."""

from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement

from tests.public_api_support import _project_metadata, _repo_root


def _requirement_file_entries(path: Path) -> tuple[list[str], list[Requirement]]:
    """Split a pip requirements file into include lines and requirements."""
    includes: list[str] = []
    requirements: list[Requirement] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            includes.append(line)
        else:
            requirements.append(Requirement(line))
    return includes, requirements


def test_requirements_files_mirror_pyproject_dependencies() -> None:
    """The pip requirements files offered in the README mirror pyproject.

    Each file lists exactly the names pyproject declares, with the same
    extras and markers, and keeps every version clause pyproject states;
    a file may only add a stricter floor to an unpinned name.
    """
    root = _repo_root()
    project = _project_metadata()["project"]
    profiles: tuple[tuple[str, list[str], list[str]], ...] = (
        ("requirements.txt", project["dependencies"], []),
        ("requirements-dev.txt", project["optional-dependencies"]["dev"], ["-r requirements.txt"]),
    )
    for file_name, declared, expected_includes in profiles:
        includes, listed = _requirement_file_entries(root / file_name)
        assert includes == expected_includes, file_name
        mirror = {requirement.name: requirement for requirement in listed}
        source = {requirement.name: requirement for requirement in map(Requirement, declared)}
        assert len(mirror) == len(listed), f"{file_name} lists a name twice"
        assert sorted(mirror) == sorted(source), file_name
        for name, requirement in source.items():
            copy = mirror[name]
            assert copy.extras == requirement.extras, f"{file_name}: {name}"
            assert str(copy.marker) == str(requirement.marker), f"{file_name}: {name}"
            clauses = {str(clause) for clause in requirement.specifier}
            copy_clauses = {str(clause) for clause in copy.specifier}
            assert clauses <= copy_clauses, f"{file_name}: {name}"
            if any(clause.operator == "==" for clause in requirement.specifier):
                assert copy_clauses == clauses, f"{file_name}: {name}"
