#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — install-profile audit tool

"""Audit SC-NeuroCore install-profile dependency and packaging boundaries."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import venv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:  # pragma: no cover - covered by Python-version matrix.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

INSTALL_PROFILE_AUDIT_SCHEMA_VERSION = "sc-neurocore.install-profile-audit.v1"
HEAVY_BASE_DEPENDENCIES = frozenset(
    {
        "cupy",
        "cupy-cuda12x",
        "fastapi",
        "httpx",
        "jax",
        "jaxlib",
        "lava",
        "matplotlib",
        "mpi4py",
        "networkx",
        "nir",
        "onnx",
        "pennylane",
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "torch",
        "uvicorn",
    }
)
POLYGLOT_PACKAGE_PREFIXES = ("accel/julia/", "accel/go/", "accel/mojo/")
STATIC_PRIMITIVE_PATTERN = "hdl/primitives/*.v"
EXPECTED_STATIC_PRIMITIVES = (
    "hdl/primitives/sc_bitstream_encoder.v",
    "hdl/primitives/sc_bitstream_synapse.v",
    "hdl/primitives/sc_dense_layer_core.v",
    "hdl/primitives/sc_dotproduct_to_current.v",
    "hdl/primitives/sc_firing_rate_bank.v",
    "hdl/primitives/sc_lif_neuron.v",
)
#: A dependency name followed by its version specifier, as ``pyproject.toml``
#: writes it. The specifier may be empty for an unconstrained requirement.
_REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?P<specifier>.*)$")

#: The only environment marker the conda recipe can express, as a selector
#: comment. A requirement carrying any other marker cannot be translated and is
#: reported rather than silently dropped from the expected recipe.
_PYTHON_VERSION_MARKER = re.compile(
    r"^python_version\s*(?P<operator>[<>=!]+)\s*[\'\"](?P<version>\d+\.\d+)[\'\"]$"
)


class InstallProfileAuditError(RuntimeError):
    """Raised when project metadata cannot be translated into a conda recipe."""


def conda_run_dependencies(requires_python: str, dependencies: Sequence[str]) -> tuple[str, ...]:
    """Return the conda ``run:`` list that the project metadata implies.

    The recipe is checked against this derivation rather than against a copy of
    itself, so a dependency added to ``pyproject.toml`` and nowhere else is a
    reported misalignment instead of two stale statements agreeing.

    Parameters
    ----------
    requires_python : str
        The ``project.requires-python`` specifier, for example ``">=3.10"``.
    dependencies : Sequence[str]
        The ``project.dependencies`` requirements, in the order authored.

    Returns
    -------
    tuple of str
        The interpreter row followed by one row per requirement, formatted the
        way ``conda/meta.yaml`` writes them.

    Raises
    ------
    InstallProfileAuditError
        If a requirement is unparsable, or carries an environment marker that
        a conda selector cannot express.
    """
    rows = [f"python {requires_python.strip()}"]
    rows.extend(_conda_requirement(requirement) for requirement in dependencies)
    return tuple(rows)


def _conda_requirement(requirement: str) -> str:
    """Return one ``pyproject.toml`` requirement as a conda recipe row."""
    specification, _, marker = requirement.partition(";")
    match = _REQUIREMENT.match(specification.strip())
    if match is None:
        raise InstallProfileAuditError(f"Cannot parse base requirement {requirement!r}")
    name = match["name"].strip().lower().replace("_", "-")
    specifier = match["specifier"].strip()
    row = f"{name} {specifier}".strip()
    if marker.strip():
        row = f"{row}{_conda_selector(marker)}"
    return row


def _conda_selector(marker: str) -> str:
    """Return the conda selector comment for a Python-version marker."""
    match = _PYTHON_VERSION_MARKER.match(marker.strip())
    if match is None:
        raise InstallProfileAuditError(
            f"Environment marker {marker.strip()!r} has no conda selector equivalent"
        )
    version = match["version"].replace(".", "")
    return f"  # [py{match['operator']}{version}]"


EXPECTED_HUB_DEPENDENCY_MIRRORS = (
    "mirrors/wheelhouse",
    "mirrors/huggingface",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--measure-install",
        action="store_true",
        help="Create a temporary venv and time a local no-dependency wheel install.",
    )
    return parser


def build_install_profile_audit(
    repo: Path,
    *,
    measure_install: bool = False,
) -> dict[str, Any]:
    repo = repo.resolve()
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    dependencies = list(project.get("dependencies", []))
    extras = dict(project.get("optional-dependencies", {}))
    package_data = list(
        pyproject.get("tool", {})
        .get("setuptools", {})
        .get("package-data", {})
        .get("sc_neurocore", [])
    )
    dependency_names = {_normalise_requirement_name(dep) for dep in dependencies}
    heavy_base = sorted(name for name in dependency_names if name in HEAVY_BASE_DEPENDENCIES)
    polyglot_data = sorted(
        item for item in package_data if item.startswith(POLYGLOT_PACKAGE_PREFIXES)
    )
    offline_hardware_profile = _build_offline_hardware_profile(
        repo=repo,
        project_version=project["version"],
        package_data=package_data,
        requires_python=str(project["requires-python"]),
        dependencies=dependencies,
    )
    install_measurement = (
        _measure_local_base_install(repo) if measure_install else {"measured": False}
    )
    passed = (
        not heavy_base
        and not polyglot_data
        and not offline_hardware_profile["missing_static_primitives"]
        and offline_hardware_profile["conda_recipe_aligned"]
        and offline_hardware_profile["docker_wheel_build_covers_static_primitives"]
        and offline_hardware_profile["hub_offline_mirror_gate"]
    )
    return {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": INSTALL_PROFILE_AUDIT_SCHEMA_VERSION,
        "project": {
            "name": project["name"],
            "version": project["version"],
        },
        "base_dependencies": dependencies,
        "base_dependency_count": len(dependencies),
        "heavy_dependencies_in_base": heavy_base,
        "optional_extras": sorted(extras),
        "optional_extra_count": len(extras),
        "packaged_data": package_data,
        "polyglot_research_sources_in_wheel": polyglot_data,
        "offline_hardware_profile": offline_hardware_profile,
        "install_measurement": install_measurement,
        "passed": passed and bool(install_measurement.get("passed", True)),
    }


def _normalise_requirement_name(requirement: str) -> str:
    name = requirement.split(";", maxsplit=1)[0].strip()
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        name = name.split(marker, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _build_offline_hardware_profile(
    *,
    repo: Path,
    project_version: str,
    package_data: list[str],
    requires_python: str,
    dependencies: Sequence[str],
) -> dict[str, Any]:
    matched_package_data = _matched_package_data(repo, package_data)
    missing_static_primitives = sorted(
        primitive
        for primitive in EXPECTED_STATIC_PRIMITIVES
        if primitive not in matched_package_data
    )
    conda_recipe = _read_conda_recipe(repo)
    expected_run = conda_run_dependencies(requires_python, dependencies)
    recorded_run = list(conda_recipe["run_dependencies"])
    # Reported as two ordered lists rather than one boolean: a recipe that has
    # fallen behind the project metadata should name the rows it is missing.
    missing_run = [row for row in expected_run if row not in recorded_run]
    unexpected_run = [row for row in recorded_run if row not in expected_run]
    conda_recipe_aligned = (
        conda_recipe["version"] == project_version
        and not missing_run
        and not unexpected_run
        and "sc_neurocore.hdl.resources" in conda_recipe["test_imports"]
        and any(
            "list_baseline_primitive_rtl" in command for command in conda_recipe["test_commands"]
        )
    )
    docker_wheel_build_covers_static_primitives = _docker_wheel_build_covers_static_primitives(repo)
    hub_profile = _read_hub_airgap_profile(repo)
    if STATIC_PRIMITIVE_PATTERN not in package_data:
        missing_static_primitives = sorted(
            set(missing_static_primitives) | set(EXPECTED_STATIC_PRIMITIVES)
        )
    return {
        "profile": "hdl-offline",
        "pip_install": 'pip install "sc-neurocore[hdl]"',
        "docker_build_arg": "INSTALL_EXTRAS=hdl",
        "vivado_required_for_baseline": False,
        "static_primitive_pattern": STATIC_PRIMITIVE_PATTERN,
        "expected_static_primitives": list(EXPECTED_STATIC_PRIMITIVES),
        "missing_static_primitives": missing_static_primitives,
        "conda_run_dependencies_expected": list(expected_run),
        "conda_run_dependencies_recorded": recorded_run,
        "conda_run_dependencies_missing": missing_run,
        "conda_run_dependencies_unexpected": unexpected_run,
        "conda_recipe_aligned": conda_recipe_aligned,
        "docker_wheel_build_covers_static_primitives": docker_wheel_build_covers_static_primitives,
        "hub_dependency_mirrors": hub_profile["hub_dependency_mirrors"],
        "hub_air_gapped_contract": hub_profile["hub_air_gapped_contract"],
        "hub_offline_mirror_gate": hub_profile["hub_offline_mirror_gate"],
    }


def _read_hub_airgap_profile(repo: Path) -> dict[str, Any]:
    bundle = (repo / "src" / "sc_neurocore" / "hub" / "bundle.py").read_text(encoding="utf-8")
    mirrors = EXPECTED_HUB_DEPENDENCY_MIRRORS
    has_offline_gate = (
        "offline: bool = True" in bundle
        and "offline mode requires at least one dependency_mirror_dirs entry" in bundle
        and all(mirror in bundle for mirror in EXPECTED_HUB_DEPENDENCY_MIRRORS)
    )
    gate = has_offline_gate and mirrors == EXPECTED_HUB_DEPENDENCY_MIRRORS
    return {
        "hub_dependency_mirrors": list(mirrors),
        "hub_air_gapped_contract": {
            "requires_local_dependency_mirrors": has_offline_gate,
            "dependency_mirror_dirs": list(mirrors),
        },
        "hub_offline_mirror_gate": gate,
    }


def _matched_package_data(repo: Path, package_data: list[str]) -> set[str]:
    package_root = repo / "src" / "sc_neurocore"
    matched: set[str] = set()
    for pattern in package_data:
        matched.update(
            str(path.relative_to(package_root))
            for path in package_root.glob(pattern)
            if path.is_file()
        )
    return matched


def _docker_wheel_build_covers_static_primitives(repo: Path) -> bool:
    dockerfile = repo / "deploy" / "Dockerfile"
    if not dockerfile.is_file():
        return False
    text = dockerfile.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "COPY src/ ./src/",
            "COPY requirements/hdl.txt ./requirements/hdl.txt",
            "INSTALL_EXTRAS",
            "requirements/hdl.txt",
            "python -m build --wheel",
            "list_baseline_primitive_rtl",
        )
    )


def _read_conda_recipe(repo: Path) -> dict[str, Any]:
    recipe = repo / "conda" / "meta.yaml"
    lines = recipe.read_text(encoding="utf-8").splitlines()
    return {
        "version": _read_conda_version(lines),
        "run_dependencies": _read_nested_list(lines, ("requirements:", "run:")),
        "test_imports": _read_nested_list(lines, ("test:", "imports:")),
        "test_commands": _read_nested_list(lines, ("test:", "commands:")),
    }


def _read_conda_version(lines: list[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("{% set version = "):
            return stripped.split('"', maxsplit=2)[1]
    return ""


def _read_nested_list(lines: list[str], path: tuple[str, str]) -> list[str]:
    in_parent = False
    in_child = False
    values: list[str] = []
    parent, child = path
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped == parent:
            in_parent = True
            in_child = False
            continue
        if in_parent and indent == 0 and stripped.endswith(":"):
            break
        if in_parent and stripped == child:
            in_child = True
            continue
        if in_child:
            if stripped.startswith("- "):
                values.append(stripped[2:])
                continue
            if stripped and indent <= 2:
                break
    return values


def _tail_lines(text: str, *, limit: int = 12) -> list[str]:
    return text.strip().splitlines()[-limit:]


def _measure_local_base_install(repo: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="scn-install-profile-") as tmp:
        venv_dir = Path(tmp) / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        start = time.perf_counter()
        result = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                str(repo),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        elapsed = time.perf_counter() - start
        smoke = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import sc_neurocore; "
                    "print(sc_neurocore.__version__); "
                    "print(len(sc_neurocore.__all__))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        package_list = subprocess.run(
            [str(python), "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        installed_packages = _parse_pip_list(package_list.stdout)
        installed_package_names = {name for name, _version in installed_packages}
        heavy_installed = sorted(
            name for name in installed_package_names if name in HEAVY_BASE_DEPENDENCIES
        )
        return {
            "measured": True,
            "command": "python -m pip install <repo>",
            "elapsed_seconds": round(elapsed, 3),
            "install_returncode": result.returncode,
            "install_stdout_tail": _tail_lines(result.stdout),
            "install_stderr_tail": _tail_lines(result.stderr),
            "smoke_returncode": smoke.returncode,
            "smoke_stdout": smoke.stdout.strip().splitlines(),
            "smoke_stderr_tail": _tail_lines(smoke.stderr),
            "pip_list_returncode": package_list.returncode,
            "pip_list_stderr_tail": _tail_lines(package_list.stderr),
            "installed_package_count": len(installed_packages),
            "installed_packages": [
                {"name": name, "version": version} for name, version in installed_packages
            ],
            "heavy_optional_packages_installed": heavy_installed,
            "passed": (
                result.returncode == 0
                and smoke.returncode == 0
                and package_list.returncode == 0
                and not heavy_installed
            ),
        }


_measure_local_no_deps_install = _measure_local_base_install


def _parse_pip_list(payload: str) -> list[tuple[str, str]]:
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError:
        return []
    packages: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        version = row.get("version")
        if isinstance(name, str) and isinstance(version, str):
            packages.append((name.lower().replace("_", "-"), version))
    return sorted(packages)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_install_profile_audit(args.repo, measure_install=args.measure_install)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
