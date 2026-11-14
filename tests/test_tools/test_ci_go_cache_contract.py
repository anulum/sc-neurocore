# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go toolchain cache dependency contract

"""The CI workflow's Go cache, its accelerator builds, and the backends it requires."""

from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_ci_setup_go_caches_tracked_backend_manifests() -> None:
    """The Go cache keys on the tracked module manifests the backends build from."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    setup_go = next(step for step in steps if step.get("uses", "").startswith("actions/setup-go@"))
    dependency_paths = setup_go["with"]["cache-dependency-path"].splitlines()

    assert dependency_paths == [
        "src/sc_neurocore/accel/go/go.mod",
        "src/sc_neurocore/accel/go/services/hil_debugger/go.sum",
    ]
    assert all((ROOT / relative).is_file() for relative in dependency_paths)
    assert all(relative in _tracked_files() for relative in dependency_paths)


def test_ci_builds_rk4_neuron_parity_backends() -> None:
    """The RK4 parity backends are built for Go and Mojo, through the explicit loop."""
    workflow_path = ROOT / ".github/workflows/ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    build_step = next(
        step for step in steps if step.get("name") == "Build accelerator backends (Go + Mojo)"
    )
    run_text = build_step["run"]

    assert "partition rk4_neurons" in run_text
    assert 'go_cshared "src/sc_neurocore/accel/go/${model}" "lib${model}.so" .' in run_text
    assert '"$MOJO_BIN" build --emit shared-lib --target-cpu x86-64-v3' in run_text
    assert '--mojo-command "$MOJO_BIN"' in run_text


#: The one c-shared build of a workflow step: into a temporary folder, from which only
#: the library is copied beside its source. cgo also writes a C header beside its
#: output, and the tracked header beside the source is a committed ABI that tests pin.
OUT_OF_TREE_BUILD = (
    'go build -buildmode=c-shared -o "$out/$lib" "$@"',
    'cp "$out/$lib" "$dir/$lib"',
)


def _assert_go_builds_out_of_tree(run_text: str) -> None:
    """Fail when a step builds a Go library anywhere but through its out-of-tree function.

    Parameters
    ----------
    run_text : str
        The shell text of a workflow step.
    """
    assert run_text.count("go build -buildmode=c-shared") == 1
    for line in OUT_OF_TREE_BUILD:
        assert line in run_text


def test_ci_builds_go_libraries_out_of_the_source_tree() -> None:
    """No CI build writes cgo's header over the committed one beside a source."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    build_step = next(
        step for step in steps if step.get("name") == "Build accelerator backends (Go + Mojo)"
    )
    _assert_go_builds_out_of_tree(build_step["run"])
    others = [
        step["run"]
        for step in steps
        if step is not build_step and "go build -buildmode=c-shared" in step.get("run", "")
    ]
    assert others == []


def test_ci_builds_both_wilson_hr_runtime_identities() -> None:
    """Both Wilson runtime identities, cortical and resetting SC, are built for Go and Mojo."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    build_step = next(
        step for step in steps if step.get("name") == "Build accelerator backends (Go + Mojo)"
    )
    run_text = build_step["run"]

    for path in (
        "src/sc_neurocore/accel/go/neurons/wilson_hr",
        "src/sc_neurocore/accel/go/neurons/sc_resetting_wilson_hr",
        "src/sc_neurocore/accel/mojo/neurons/wilson_hr.mojo",
        "src/sc_neurocore/accel/mojo/neurons/sc_resetting_wilson_hr.mojo",
    ):
        assert path in run_text
    assert "libwilsonhr.so" in run_text
    assert "libsc_resetting_wilson_hr.so" in run_text


def test_ci_requires_both_rulkov_event_identities() -> None:
    """CI must execute every compiled lane for both Rulkov event contracts."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    build_step = next(
        step for step in steps if step.get("name") == "Build accelerator backends (Go + Mojo)"
    )
    required_line = next(
        line.strip()
        for line in build_step["run"].splitlines()
        if line.strip().startswith("REQUIRED=")
    )
    required = set(required_line.removeprefix("REQUIRED=").split(","))

    assert {"rulkov_map", "sc_upward_crossing_rulkov_map"} <= required


def _tracked_files() -> set[str]:
    """Return repository-relative paths recorded in the Git index."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return set(completed.stdout.split("\0"))
