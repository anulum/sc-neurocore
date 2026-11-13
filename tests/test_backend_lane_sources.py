# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The sources and build inputs of a backend lane as subjects of its facet

"""Bind how a lane's sources and build inputs become the subjects of its facet.

A native lane's facet requires ``native-backend`` subjects: the files that
implement the model in that lane, as its descriptor lists them; the manifests,
lock files, build configurations and tool pins that fix the lane's build; and,
for the Rust lane, every tracked file of the folders its library is built from.
A source becomes a subject only when it is a tracked file spelled as Git spells
it; any other source makes the lane's facet ``unavailable`` and names the path,
so that a receipt never binds a file another checkout would not have.
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import tomli_w

from sc_neurocore.neurons import readiness
from sc_neurocore.neurons.evidence_references import sha256_file
from sc_neurocore.neurons.facet_receipts import FACETS
from sc_neurocore.neurons.readiness import (
    LANE_BUILD_INPUTS,
    LANE_BUILD_TREES,
    derive_subjects,
    lane_source_problems,
    verify_model,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DESCRIPTOR = "src/sc_neurocore/neurons/model_descriptors/LapicqueNeuron.toml"
_SOURCES = ("engine/src/neuron/lapicque.rs", "engine/src/bindings/lapicque.rs")
# The six spellings that pass the descriptor parser and name no file as Git spells it.
_MISSPELLINGS = (
    "engine\\src\\neuron\\lapicque.rs",
    "./engine/src/neuron/lapicque.rs",
    "engine//src/neuron/lapicque.rs",
    "   ",
    "engine/src/neuron/",
    "engine/src/neuron/lapi\ncque.rs",
)
# Files of the folders the Rust lane binds (the vendored crate and the engine's Python
# package), written into a test root.
_VENDORED = (
    "bridge/vendor/rayon-core/Cargo.toml",
    "bridge/vendor/rayon-core/src/lib.rs",
    "bridge/sc_neurocore_engine/__init__.py",
)


def _build(lane: str) -> tuple[str, ...]:
    """Return the build files a test root holds for a lane: its inputs and its vendored files."""
    return (*LANE_BUILD_INPUTS[lane], *(_VENDORED if lane == "rust" else ()))


def _payload(lane: str, sources: list[str]) -> dict[str, Any]:
    """Return the shipped Lapicque descriptor with ``sources`` on one implemented lane."""
    payload = tomllib.loads((_REPO_ROOT / _DESCRIPTOR).read_text(encoding="utf-8"))
    payload["backends"][lane] = {"status": "implemented", "parity": "exact", "sources": sources}
    return payload


def _root(tmp_path: Path, lane: str, sources: list[str]) -> Path:
    """Write a repository root with the descriptor, the lane's sources and build inputs."""
    descriptor = tmp_path / _DESCRIPTOR
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text(tomli_w.dumps(_payload(lane, sources)), encoding="utf-8")
    for relative in (*_SOURCES, *_build(lane)):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"// {relative}\n", encoding="utf-8")
    return tmp_path


def _native(subjects: tuple[Any, ...]) -> dict[str, str]:
    """Return the ``native-backend`` subjects as path to digest."""
    return {s.path: s.sha256 for s in subjects if s.kind == "native-backend"}


@pytest.fixture
def git_root(tmp_path: Path) -> Iterator[Path]:
    """Yield a root that is a Git checkout whose index holds the sources and build inputs."""
    git = shutil.which("git")
    assert git is not None, "these cases need Git; a skip would hide the rule they hold"
    root = _root(tmp_path, "rust", list(_SOURCES))
    subprocess.run([git, "init", "-q", str(root)], check=True, timeout=30)
    subprocess.run([git, "-C", str(root), "add", "-A"], check=True, timeout=30)
    yield root


@pytest.mark.parametrize("lane", ["rust", "rust_safety", "julia", "go", "mojo"])
def test_a_lane_s_sources_and_build_inputs_are_its_native_subjects(
    tmp_path: Path, lane: str
) -> None:
    """Two sources and the lane's build inputs, each with the digest of its file.

    Parameters
    ----------
    lane : str
        A native lane; the descriptor's other lanes list no source.
    """
    root = _root(tmp_path, lane, list(_SOURCES))
    subjects = derive_subjects("LapicqueNeuron", f"backend:{lane}", repo_root=root)
    expected = {path: sha256_file(root / path) for path in (*_SOURCES, *_build(lane))}
    assert _native(subjects) == expected


def test_editing_one_source_changes_exactly_its_digest(tmp_path: Path) -> None:
    """The other source and the build inputs keep their digests."""
    root = _root(tmp_path, "rust", list(_SOURCES))
    before = _native(derive_subjects("LapicqueNeuron", "backend:rust", repo_root=root))
    (root / _SOURCES[0]).write_text("// changed\n", encoding="utf-8")
    after = _native(derive_subjects("LapicqueNeuron", "backend:rust", repo_root=root))
    assert {path for path in before if before[path] != after[path]} == {_SOURCES[0]}
    assert set(before) == set(after)


@pytest.mark.parametrize("spelling", _MISSPELLINGS)
def test_a_source_not_spelled_as_git_spells_it_is_no_subject(tmp_path: Path, spelling: str) -> None:
    """Without Git the normal form decides: each of the six spellings is refused.

    Parameters
    ----------
    spelling : str
        A source that passes the descriptor parser and is not a normal path.
    """
    root = _root(tmp_path, "rust", [spelling])
    misspelt = root / spelling
    if not misspelt.exists():
        # A file under the misspelt name exists, so only the spelling can refuse it.
        misspelt.parent.mkdir(parents=True, exist_ok=True)
        misspelt.write_text("// a file under a misspelt name\n", encoding="utf-8")
    assert lane_source_problems((spelling,), root) == (
        f"source is not a tracked file: {spelling!r}",
    )
    assert set(_native(derive_subjects("LapicqueNeuron", "backend:rust", repo_root=root))) == set(
        _build("rust")
    )


@pytest.mark.parametrize("spelling", _MISSPELLINGS)
def test_in_a_git_checkout_only_the_index_spelling_is_a_source(
    git_root: Path, spelling: str
) -> None:
    """With Git the index decides: the six spellings are refused and the exact one is accepted.

    Parameters
    ----------
    git_root : Path
        A checkout whose index holds the two sources.
    spelling : str
        A source that passes the descriptor parser and is not in the index.
    """
    assert lane_source_problems((spelling, *_SOURCES), git_root) == (
        f"source is not a tracked file: {spelling!r}",
    )


def test_in_a_git_checkout_an_untracked_file_is_no_source(git_root: Path) -> None:
    """A file that exists but is not in the index is refused like a missing one."""
    loose = git_root / "engine/src/neuron/loose.rs"
    loose.write_text("// not added\n", encoding="utf-8")
    assert lane_source_problems(("engine/src/neuron/loose.rs",), git_root) == (
        "source is not a tracked file: 'engine/src/neuron/loose.rs'",
    )


def test_a_tracked_source_deleted_from_the_tree_is_refused(git_root: Path) -> None:
    """The index alone does not make a subject: the file must be there to be digested."""
    (git_root / _SOURCES[1]).unlink()
    assert lane_source_problems(_SOURCES, git_root) == (
        f"source is not a tracked file: {_SOURCES[1]!r}",
    )
    subjects = derive_subjects("LapicqueNeuron", "backend:rust", repo_root=git_root)
    assert _SOURCES[1] not in _native(subjects)
    assert _SOURCES[0] in _native(subjects)


def test_a_checkout_without_git_on_the_path_accepts_no_source(
    git_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In a checkout the index decides; without Git to read it, every source is unchecked and says why."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert lane_source_problems(_SOURCES, git_root) == tuple(
        f"source cannot be checked against the index (Git is not on the path): {source!r}"
        for source in _SOURCES
    )


@pytest.mark.parametrize(
    "failure", [OSError("git cannot start"), subprocess.TimeoutExpired(["git"], 60)]
)
def test_a_checkout_where_git_does_not_answer_accepts_no_source(
    git_root: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """A Git that cannot start or does not finish lists nothing; every source is unchecked and says why.

    Parameters
    ----------
    git_root : Path
        A checkout whose index holds the two sources.
    monkeypatch : pytest.MonkeyPatch
        Replaces the process call.
    failure : Exception
        What the process call raises.
    """

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(subprocess, "run", refuse)
    assert lane_source_problems(_SOURCES, git_root) == tuple(
        f"source cannot be checked against the index (Git did not answer): {source!r}"
        for source in _SOURCES
    )
    assert set(_native(derive_subjects("LapicqueNeuron", "backend:rust", repo_root=git_root))) == (
        set(LANE_BUILD_INPUTS["rust"])
    )


def test_a_damaged_checkout_accepts_no_source(tmp_path: Path) -> None:
    """A ``.git`` entry that Git cannot read makes the listing fail; nothing is tracked.

    No source and no vendored file is bound; the build inputs, named by the
    repository itself, still are.
    """
    root = _root(tmp_path, "rust", list(_SOURCES))
    (root / ".git").write_text("gitdir: missing-directory\n", encoding="utf-8")
    assert lane_source_problems(_SOURCES[:1], root) == (
        f"source cannot be checked against the index (Git could not read the checkout): "
        f"{_SOURCES[0]!r}",
    )
    subjects = derive_subjects("LapicqueNeuron", "backend:rust", repo_root=root)
    assert set(_native(subjects)) == set(LANE_BUILD_INPUTS["rust"])


def test_a_root_without_the_descriptor_binds_the_build_inputs_only(tmp_path: Path) -> None:
    """With no descriptor there are no sources to read; the lane's build inputs remain."""
    root = _root(tmp_path, "go", ["src/sc_neurocore/accel/go/lapicque.go"])
    (root / _DESCRIPTOR).unlink()
    subjects = derive_subjects("LapicqueNeuron", "backend:go", repo_root=root)
    assert set(_native(subjects)) == set(_build("go"))


def test_a_missing_source_is_named(tmp_path: Path) -> None:
    """A source spelled correctly that names no file is refused with its path."""
    root = _root(tmp_path, "go", ["src/sc_neurocore/accel/go/lapicque.go"])
    assert lane_source_problems(("src/sc_neurocore/accel/go/lapicque.go",), root) == (
        "source is not a tracked file: 'src/sc_neurocore/accel/go/lapicque.go'",
    )


def test_refused_sources_are_named_in_their_written_order(tmp_path: Path) -> None:
    """Two refused sources come back in the order the descriptor lists them, the accepted one omitted."""
    root = _root(tmp_path, "go", [])
    listed = ("src/z_missing.go", _SOURCES[0], "src/a_missing.go")
    assert lane_source_problems(listed, root) == (
        "source is not a tracked file: 'src/z_missing.go'",
        "source is not a tracked file: 'src/a_missing.go'",
    )


def test_the_reference_lane_and_the_other_axes_derive_no_native_subject(tmp_path: Path) -> None:
    """Only a native lane's facet takes sources and build inputs."""
    root = _root(tmp_path, "rust", list(_SOURCES))
    for spec in FACETS:
        if spec.name.startswith("backend:") and spec.name != "backend:python":
            continue
        subjects = derive_subjects("LapicqueNeuron", spec.name, repo_root=root)
        assert _native(subjects) == {}, spec.name


def test_a_lane_without_sources_still_binds_its_build_inputs(tmp_path: Path) -> None:
    """The build inputs fix what a lane computes even before its sources are listed."""
    root = _root(tmp_path, "mojo", [])
    subjects = derive_subjects("LapicqueNeuron", "backend:mojo", repo_root=root)
    assert set(_native(subjects)) == set(_build("mojo"))


def test_a_lane_with_a_refused_source_is_unavailable_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verifier reports the descriptor's error before any receipt is judged."""
    root = _root(tmp_path, "julia", ["engine//src/neuron/lapicque.rs"])
    payload = _payload("julia", ["engine//src/neuron/lapicque.rs"])
    monkeypatch.setattr(readiness, "load_descriptor_payload", lambda _name: payload)
    record = verify_model("LapicqueNeuron", profile="lapicque", repo_root=root)
    facet = record.facet("backend:julia")
    assert facet.status == "unavailable"
    assert facet.problems == ("source is not a tracked file: 'engine//src/neuron/lapicque.rs'",)
    assert record.facet("backend:rust").status != "unavailable"


def test_one_derivation_reads_the_index_once(
    git_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Rust lane's sources and both its folders are judged from one listing of the index.

    A lane with neither sources nor folders reads no index at all.
    """
    calls: list[object] = []
    run = subprocess.run

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", counted)
    subjects = _native(derive_subjects("LapicqueNeuron", "backend:rust", repo_root=git_root))
    assert set(_SOURCES) | set(_VENDORED) <= set(subjects)
    assert len(calls) == 1
    calls.clear()
    derive_subjects("LapicqueNeuron", "backend:go", repo_root=git_root)
    assert calls == []


def test_the_folders_of_the_rust_lane_are_named_and_tracked() -> None:
    """The bridge's and the root configuration replace rayon-core by the vendored copy.

    The engine wheel is built from ``bridge/``; its Python package and the vendored
    crate are both compiled or packed into the library the lane loads.
    """
    assert LANE_BUILD_TREES == {"rust": ("bridge/sc_neurocore_engine", "bridge/vendor/rayon-core")}
    root_configuration = (_REPO_ROOT / ".cargo/config.toml").read_text(encoding="utf-8")
    bridge_configuration = (_REPO_ROOT / "bridge/.cargo/config.toml").read_text(encoding="utf-8")
    assert 'paths = ["bridge/vendor/rayon-core"]' in root_configuration
    assert 'paths = ["vendor/rayon-core"]' in bridge_configuration
    listed = _git_files("bridge/vendor/rayon-core", "bridge/sc_neurocore_engine")
    assert {
        "bridge/vendor/rayon-core/Cargo.toml",
        "bridge/sc_neurocore_engine/__init__.py",
    } <= listed


def _git_files(*patterns: str) -> set[str]:
    """Return the tracked files that match the patterns, ``**`` matching zero or more folders.

    Git's ``:(glob)`` magic gives a pattern the meaning a workflow's ``hashFiles``
    gives it; a plain pathspec would need a folder where ``**/`` stands, and would
    so read the cache key more narrowly than the key itself.
    """
    assert (_REPO_ROOT / ".git").exists(), "these cases read the repository's index"
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", *(f":(glob){pattern}" for pattern in patterns)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    return {path for path in listing.split("\0") if path}


# Named by the cache key and in no library: the bridge's top-level package file. The engine
# wheel packs only the package its module name names (``python-source = "."`` and
# ``module-name = "sc_neurocore_engine.sc_neurocore_engine"`` in ``bridge/pyproject.toml``),
# so no lane loads this file; an edit of it changes no run.
_NAMED_BY_THE_KEY_OUTSIDE_THE_LIBRARY = frozenset({"bridge/__init__.py"})


def _engine_wheel_cache_patterns() -> list[str]:
    """Return the path patterns of the engine wheel's exact-source cache key in the main workflow."""
    workflow = (_REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    keys = [
        line
        for line in workflow.splitlines()
        if "-engine-v1-release-" in line and "hashFiles(" in line
    ]
    assert len(keys) == 1, keys
    inside = keys[0].split("hashFiles(", 1)[1].rsplit(")", 1)[0]
    return [part.strip().strip("'") for part in inside.split(",")]


def test_the_rust_lane_binds_what_the_engine_wheel_cache_key_names() -> None:
    """Every tracked file the wheel's cache key names is bound, save one named file outside the wheel.

    The repository's own key for reusing an engine wheel lists what decides that build,
    read with the key's own meaning of ``**``. The lane's sources (``engine/src``) are
    bound through the descriptors instead. Every build input is named by the key.
    """
    patterns = _engine_wheel_cache_patterns()
    named = _git_files(*[pattern for pattern in patterns if not pattern.startswith("engine/src/")])
    bound = set(LANE_BUILD_INPUTS["rust"]) | {
        path for tree in LANE_BUILD_TREES["rust"] for path in _git_files(tree)
    }
    assert named - bound == _NAMED_BY_THE_KEY_OUTSIDE_THE_LIBRARY, sorted(named - bound)
    assert set(LANE_BUILD_INPUTS["rust"]) <= named, sorted(set(LANE_BUILD_INPUTS["rust"]) - named)


def test_the_engine_wheel_packs_only_the_package_the_lane_binds() -> None:
    """The file left unbound lies outside the one package the wheel's configuration packs."""
    maturin = tomllib.loads((_REPO_ROOT / "bridge/pyproject.toml").read_text(encoding="utf-8"))[
        "tool"
    ]["maturin"]
    assert maturin["python-source"] == "."
    package = "bridge/" + maturin["module-name"].split(".")[0]
    assert package in LANE_BUILD_TREES["rust"]
    for path in _NAMED_BY_THE_KEY_OUTSIDE_THE_LIBRARY:
        assert not path.startswith(package + "/"), path


def test_every_file_of_the_vendored_crate_is_a_file_subject(tmp_path: Path) -> None:
    """Each file is bound with its own digest; editing one changes exactly that subject."""
    root = _root(tmp_path, "rust", [])
    before = _native(derive_subjects("LapicqueNeuron", "backend:rust", repo_root=root))
    assert {path: before[path] for path in _VENDORED} == {
        path: sha256_file(root / path) for path in _VENDORED
    }
    (root / _VENDORED[1]).write_text("// changed\n", encoding="utf-8")
    after = _native(derive_subjects("LapicqueNeuron", "backend:rust", repo_root=root))
    assert {path for path in before if before[path] != after[path]} == {_VENDORED[1]}


def test_in_a_git_checkout_only_tracked_vendored_files_are_bound(git_root: Path) -> None:
    """A file left in the vendored folder but not in the index is not a subject."""
    (git_root / "bridge/vendor/rayon-core/target.log").write_text(
        "build output\n", encoding="utf-8"
    )
    subjects = _native(derive_subjects("LapicqueNeuron", "backend:rust", repo_root=git_root))
    assert set(_VENDORED) <= set(subjects)
    assert "bridge/vendor/rayon-core/target.log" not in subjects


def test_only_the_rust_lane_binds_the_vendored_crate(tmp_path: Path) -> None:
    """The safety crate does not depend on rayon-core, so its facet binds no vendored file."""
    root = _root(tmp_path, "rust", [])
    for relative in LANE_BUILD_INPUTS["rust_safety"]:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(f"// {relative}\n", encoding="utf-8")
    subjects = _native(derive_subjects("LapicqueNeuron", "backend:rust_safety", repo_root=root))
    assert set(subjects) == set(LANE_BUILD_INPUTS["rust_safety"])


def test_no_shipped_descriptor_lists_a_source_yet() -> None:
    """The part changes no status: every lane of every shipped descriptor has no sources."""
    files = sorted((_REPO_ROOT / "src/sc_neurocore/neurons/model_descriptors").glob("*.toml"))
    assert len(files) >= 185
    for path in files:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        for lane in payload.get("backends", {}).values():
            assert not isinstance(lane, dict) or "sources" not in lane, path.name
