# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia requirement gate contract

"""The Julia bridge gate skips only on genuine absence and never masks CI.

Enforced here: the helper's behaviours (present, genuinely absent,
present-but-broken, broken dependency, CI requirement), the hosted-CI
export parsed from the workflow YAML, and a pinned inventory of the
Julia-lane parity modules that must carry a direct unconditional
module-body gate call before any other statement executes Julia-lane
code. Kernel-level failures after a healthy bridge import (missing
``.jl`` files, Julia-side load errors) are deliberately outside this
gate and stay hard failures in the parity tests themselves; the
cortical-column parity module skips on a production loader flag and is
tracked as a recorded boundary rather than gated here.

Collection itself is checked too: the root conftest must leave a Julia
project that would need resolving byte-for-byte unchanged unless the run
requires Julia, and a bridge left uninitialised skips only outside CI.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tests.julia_requirement import (
    JULIA_BRIDGE_MODULE,
    require_julia,
)

_ROOT = Path(__file__).resolve().parents[1]

# Pinned inventory of Julia-lane modules gated by require_julia().
_GATED_JULIA_FILES = (
    "tests/test_adaptive_threshold_if_julia_parity.py",
    "tests/test_aihara_julia_buffer_contracts.py",
    "tests/test_alpha_julia_parity.py",
    "tests/test_gpfa_julia_parity.py",
    "tests/test_julia_rk4_neuron_parity.py",
    "tests/test_phi_estimation_julia_parity.py",
    "tests/test_resonate_and_fire_julia_parity.py",
    "tests/test_spike_stats_dimensionality_julia_parity.py",
    "tests/test_spike_stats_sorting_quality_julia_parity.py",
)


def test_require_julia_returns_the_bridge_when_present() -> None:
    """An installed and initialised bridge is returned as the imported module."""
    pytest.importorskip(JULIA_BRIDGE_MODULE)
    module = require_julia()
    assert module.__name__ == JULIA_BRIDGE_MODULE


def test_require_julia_skips_when_the_bridge_is_genuinely_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bridge that is not installed skips the calling module outside CI."""
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_JULIA", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_julia("fake_julia_bridge_that_does_not_exist")


def test_require_julia_hard_fails_on_a_present_but_broken_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bridge that is installed but fails to import is an error, never a skip."""
    package = tmp_path / "fake_julia_bridge_broken"
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise ImportError("bridge present but failed to load")', encoding="utf-8"
    )
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_JULIA", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_julia_bridge_broken", raising=False)
    with pytest.raises(ImportError, match="failed to load"):
        require_julia("fake_julia_bridge_broken")


def test_require_julia_hard_fails_on_a_broken_transitive_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing dependency of the bridge is an error, never a skip."""
    package = tmp_path / "fake_julia_bridge_broken_dep"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runtime.py").write_text(
        "import dependency_that_is_not_installed_anywhere", encoding="utf-8"
    )
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_JULIA", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_julia_bridge_broken_dep", raising=False)
    with pytest.raises(ModuleNotFoundError, match="dependency_that_is_not_installed"):
        require_julia("fake_julia_bridge_broken_dep.runtime")


def test_require_julia_hard_fails_when_ci_requires_the_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With SC_NEUROCORE_REQUIRE_JULIA=1 even a missing bridge is an error."""
    monkeypatch.setenv("SC_NEUROCORE_REQUIRE_JULIA", "1")
    with pytest.raises(ModuleNotFoundError):
        require_julia("fake_julia_bridge_that_does_not_exist")


def test_hosted_ci_exports_the_julia_requirement_at_workflow_level() -> None:
    """The main CI workflow requires Julia for every job through its top-level env."""
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert workflow["env"]["SC_NEUROCORE_REQUIRE_JULIA"] == "1"


def _unconditional_gate_line(tree: ast.Module) -> int | None:
    """First top-level statement whose value IS a require_julia call."""
    for statement in tree.body:
        value: ast.expr | None = None
        if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)):
            value = statement.value
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        named = (isinstance(func, ast.Name) and func.id == "require_julia") or (
            isinstance(func, ast.Attribute) and func.attr == "require_julia"
        )
        if named:
            return statement.lineno
    return None


def test_every_pinned_julia_module_carries_an_unconditional_gate() -> None:
    """Dominance: a direct module-body require_julia call must exist."""
    for relative in _GATED_JULIA_FILES:
        tree = ast.parse((_ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        assert _unconditional_gate_line(tree) is not None, (
            f"{relative} lost its unconditional module-body require_julia gate"
        )


def test_pinned_inventory_equals_the_derived_require_julia_set() -> None:
    """Exact set equality: silent additions fail exactly like removals."""
    derived = set()
    for path in sorted((_ROOT / "tests").rglob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _unconditional_gate_line(tree) is not None:
            derived.add(str(path.relative_to(_ROOT)))
    assert derived == set(_GATED_JULIA_FILES), (
        "require_julia-gated set drifted from the pinned inventory; "
        f"unpinned additions: {sorted(derived - set(_GATED_JULIA_FILES))}; "
        f"missing from tree: {sorted(set(_GATED_JULIA_FILES) - derived)}"
    )


def test_no_ad_hoc_juliacall_availability_probes_remain_in_tests() -> None:
    """The retired find_spec('juliacall') skip idiom must not reappear."""
    offenders = []
    for path in sorted((_ROOT / "tests").rglob("*.py")):
        if path.name in (Path(__file__).name, "julia_requirement.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "_JULIA_AVAILABLE" in text:
            offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == [], f"retired _JULIA_AVAILABLE probe reappeared in: {offenders}"


def _project_snapshot(project: Path) -> dict[str, tuple[bytes, int]]:
    """Return every file below *project* with its bytes and modification time."""
    return {
        str(path.relative_to(project)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(project.rglob("*"))
        if path.is_file()
    }


def _project_needing_resolution(tmp_path: Path, metadata: str | None = None) -> Path:
    """Create a Julia project that juliapkg would resolve again.

    Parameters
    ----------
    tmp_path : Path
        Disposable parent of the project.
    metadata : str, optional
        Raw ``meta.json`` text; by default a valid record whose Julia executable
        no longer exists, as a project resolved for another toolchain appears.

    Returns
    -------
    Path
        The project directory.
    """
    project = tmp_path / "julia_project"
    (project / "pyjuliapkg").mkdir(parents=True)
    (project / "Project.toml").write_text(
        '[deps]\nPythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"\n'
    )
    if metadata is None:
        metadata = json.dumps(
            {
                "meta_version": importlib.import_module("juliapkg.deps").META_VERSION,
                "dev": False,
                "version": "1.11.9",
                "executable": str(tmp_path / "removed" / "julia"),
                "libjulia": "",
                "bindir": "",
                "deps_files": {},
                "pkgs": [],
                "offline": True,
                "override_executable": None,
            }
        )
    (project / "pyjuliapkg" / "meta.json").write_text(metadata)
    return project


def _environment_for(project: Path) -> dict[str, str]:
    """Return this process's environment with every Julia request removed and *project* chosen."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "SC_NEUROCORE_REQUIRE_JULIA",
            "PYTHON_JULIACALL_INIT",
            "PYTHON_JULIACALL_EXE",
            "PYTHON_JULIACALL_BINDIR",
            "PYTHON_JULIACALL_PROJECT",
        }
    }
    environment.update(PYTHON_JULIAPKG_PROJECT=str(project), PYTHON_JULIAPKG_OFFLINE="yes")
    return environment


def test_collection_leaves_a_project_needing_resolution_unchanged(tmp_path: Path) -> None:
    """Collecting tests never resolves a Julia project that the run did not ask for.

    Real collection through the root conftest must leave the project's bytes,
    times and file set untouched and say why Julia stays uninitialised.
    """
    pytest.importorskip(JULIA_BRIDGE_MODULE)
    project = _project_needing_resolution(tmp_path)
    before = _project_snapshot(project)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            str(Path(__file__).relative_to(_ROOT)),
        ],
        cwd=_ROOT,
        env=_environment_for(project),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    assert f"would resolve the Julia project {project} again" in completed.stdout
    assert _project_snapshot(project) == before


_GUARD_PROBE = """
import json
import sys

from tests.julia_requirement import disable_unrequested_julia_resolution

environment = json.loads(sys.argv[1])
notice = disable_unrequested_julia_resolution(environment)
print(json.dumps({"notice": notice, "environment": environment}))
"""


def _guard_in_a_fresh_interpreter(project: Path, environment: dict[str, str]) -> dict[str, object]:
    """Run the guard where juliapkg reads *project*, never this process's own project.

    Parameters
    ----------
    project : Path
        Julia project the child's juliapkg selects.
    environment : dict
        Environment passed to the guard itself.

    Returns
    -------
    dict
        The guard's notice and the environment after the call.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _GUARD_PROBE, json.dumps(environment)],
        cwd=_ROOT,
        env=_environment_for(project),
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    result: dict[str, object] = json.loads(completed.stdout.splitlines()[-1])
    return result


@pytest.mark.parametrize(
    "metadata",
    [None, "{ damaged", json.dumps({"meta_version": 8})],
    ids=["stale", "corrupt", "partial"],
)
def test_a_project_needing_resolution_or_unreadable_disables_initialisation(
    tmp_path: Path, metadata: str | None
) -> None:
    """Without a request, a stale or unreadable project leaves JuliaCall uninitialised.

    Parameters
    ----------
    tmp_path : Path
        Disposable parent of the project.
    metadata : str or None
        Recorded metadata: a missing executable, invalid JSON or a record without its fields.
    """
    pytest.importorskip(JULIA_BRIDGE_MODULE)
    project = _project_needing_resolution(tmp_path, metadata)
    result = _guard_in_a_fresh_interpreter(project, {})
    assert result["environment"] == {"PYTHON_JULIACALL_INIT": "no"}
    assert f"would resolve the Julia project {project} again" in str(result["notice"])


@pytest.mark.parametrize(
    "requested",
    [
        {"SC_NEUROCORE_REQUIRE_JULIA": "1"},
        {"PYTHON_JULIACALL_INIT": "no"},
        {"PYTHON_JULIACALL_EXE": "/opt/julia/bin/julia", "PYTHON_JULIACALL_PROJECT": "/opt/p"},
        {"PYTHON_JULIACALL_BINDIR": "/opt/julia/bin", "PYTHON_JULIACALL_PROJECT": "/opt/p"},
    ],
)
def test_runs_that_request_or_bypass_resolution_are_left_alone(
    tmp_path: Path, requested: dict[str, str]
) -> None:
    """A run that requires Julia, disables it or names its own runtime keeps its settings.

    The guard runs against a project that would otherwise be resolved again, so
    each case passes only through its own early return.

    Parameters
    ----------
    tmp_path : Path
        Disposable parent of the project.
    requested : dict
        Environment of such a run.
    """
    pytest.importorskip(JULIA_BRIDGE_MODULE)
    result = _guard_in_a_fresh_interpreter(_project_needing_resolution(tmp_path), requested)
    assert result == {"notice": None, "environment": requested}


def test_an_uninitialised_bridge_skips_outside_ci(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bridge imported without a Julia runtime skips locally; CI receives it and fails hard."""
    package = tmp_path / "fake_julia_bridge_uninitialised"
    package.mkdir()
    (package / "__init__.py").write_text("CONFIG = {'init': False}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_julia_bridge_uninitialised", raising=False)
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_JULIA", raising=False)
    with pytest.raises(pytest.skip.Exception, match="left uninitialised"):
        require_julia("fake_julia_bridge_uninitialised")
    monkeypatch.setenv("SC_NEUROCORE_REQUIRE_JULIA", "1")
    assert require_julia("fake_julia_bridge_uninitialised").CONFIG == {"init": False}
