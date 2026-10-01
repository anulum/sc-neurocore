# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python lock profile acceptance

"""Exercise maintained lock parsing and marker/local-build refusal boundaries."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from tools.security_scan.python_dependency_profiles import (
    CPU_INDEX,
    REQUIRED_PROFILES,
    dependency_batches,
    discover_dependency_profiles,
    read_dependency_profile,
)

REPO = Path(__file__).resolve().parents[2]


def test_all_real_maintained_locks_and_additional_profiles_are_inventoried() -> None:
    """Require every live hashlock, including supported local CPU provenance."""
    profiles = discover_dependency_profiles(REPO)
    paths = {p.path for p in profiles}
    expected = {p.relative_to(REPO).as_posix() for p in (REPO / "requirements").glob("*.txt")}
    assert paths == expected - {"requirements/semgrep-overrides.txt"}
    assert set(REQUIRED_PROFILES) <= paths
    for profile in profiles:
        assert profile.sha256 == hashlib.sha256((REPO / profile.path).read_bytes()).hexdigest()
        requested = {(d.name, d.audit_version) for d in profile.dependencies}
        batches = dependency_batches(profile)
        assert {(d.name, d.audit_version) for b in batches for d in b} == requested
        assert all(len({d.name for d in b}) == len(b) for b in batches)
        assert all(d.hashes for d in profile.dependencies)
    cpu = next(p for p in profiles if p.path == "requirements/ci-torch-cpu.txt")
    torch = next(d for d in cpu.dependencies if d.name == "torch")
    assert torch.version.endswith("+cpu") and "+" not in torch.audit_version


def test_inactive_marker_and_different_versions_are_retained(tmp_path: Path) -> None:
    """Audit both versions rather than evaluating markers on the test host."""
    path = "requirements/runtime.txt"
    target = tmp_path / path
    target.parent.mkdir()
    original = (REPO / path).read_text()
    target.write_text(
        original + '\nnumpy==2.2.6; python_version < "0" --hash=sha256:' + "0" * 64 + "\n"
    )
    profile = read_dependency_profile(tmp_path, path)
    numpy = [d for d in profile.dependencies if d.name == "numpy"]
    assert len(numpy) == 2 and numpy[-1].marker == 'python_version < "0"'
    batches = dependency_batches(profile)
    assert len(batches) == 2
    assert {d.version for b in batches for d in b if d.name == "numpy"} == {
        d.version for d in numpy
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        "numpy",
        "numpy>=2",
        "numpy==2.*",
        "numpy==2.5.3",
        "-r elsewhere.txt",
        "numpy @ https://example.invalid/numpy.whl",
        "numpy==2.5.3 \\",
        "numpy==2.5.3 --hash=md5:" + "0" * 32,
        "numpy==2.5.3+private --hash=sha256:" + "0" * 64,
        "--extra-index-url https://example.invalid/simple",
    ],
)
def test_unsafe_or_unresolved_profile_input_is_refused(tmp_path: Path, text: str) -> None:
    """Reject inputs that cannot establish an exact audited identity."""
    path = tmp_path / "requirements/runtime.txt"
    path.parent.mkdir()
    path.write_text(text + "\n")
    with pytest.raises(ValueError):
        read_dependency_profile(tmp_path, "requirements/runtime.txt")


@pytest.mark.parametrize(
    "mutation",
    ["remove-runtime", "bad-override", "empty-override", "marked-override", "unhashed-extra"],
)
def test_inventory_refuses_missing_profiles_or_unreviewed_constraints(
    tmp_path: Path, mutation: str
) -> None:
    """Fail discovery using complete copied canonical locks with one defect."""
    shutil.copytree(REPO / "requirements", tmp_path / "requirements")
    if mutation == "remove-runtime":
        (tmp_path / "requirements/runtime.txt").unlink()
    elif mutation == "unhashed-extra":
        (tmp_path / "requirements/unreviewed.txt").write_text("numpy==2.5.3\n")
    else:
        text = {
            "bad-override": "click==0.1\n",
            "empty-override": "# empty\n",
            "marked-override": 'click==8.3.3; python_version < "0"\n',
        }[mutation]
        (tmp_path / "requirements/semgrep-overrides.txt").write_text(text)
    with pytest.raises(ValueError):
        discover_dependency_profiles(tmp_path)


def test_cpu_mapping_requires_exact_profile_and_official_index(tmp_path: Path) -> None:
    """An arbitrary local build cannot inherit the official CPU advisory mapping."""
    raw = (REPO / "requirements/ci-torch-cpu.txt").read_text()
    target = tmp_path / "requirements/ci-torch-cpu.txt"
    target.parent.mkdir()
    target.write_text(raw.replace(CPU_INDEX, "https://example.invalid/cpu"))
    with pytest.raises(ValueError):
        read_dependency_profile(tmp_path, "requirements/ci-torch-cpu.txt")
    (tmp_path / "requirements/runtime.txt").write_text(raw)
    with pytest.raises(ValueError):
        read_dependency_profile(tmp_path, "requirements/runtime.txt")
