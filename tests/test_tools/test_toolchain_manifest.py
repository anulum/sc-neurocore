# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Every tool declaration answers the toolchain manifest

"""The repository agrees with its toolchain manifest, and each kind of drift is refused.

The fault tests change one declaration in a copy of the files the check reads
and require the check to name it; the host tests run real executables from a
temporary search path.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from tools.toolchain_manifest import (
    MANIFEST,
    ManifestError,
    check,
    check_host,
    collect,
    load_manifest,
    main,
    workflow_declarations,
)

ROOT = Path(__file__).resolve().parents[2]
COPIED = (
    ".github",
    "tools/toolchain_manifest.toml",
    "engine/rust-toolchain.toml",
    "src/sc_neurocore/accel/mojo/pixi.toml",
    "requirements-dev.txt",
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Return a copy of every file the check reads.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the copy.

    Returns
    -------
    Path
        The copy's root.
    """
    for rel in COPIED:
        source, target = ROOT / rel, tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    for go_mod in (ROOT / "src").rglob("go.mod"):
        if any(part.startswith(".") for part in go_mod.relative_to(ROOT / "src").parts[:-1]):
            continue  # a local Go module cache, not a declaration of the repository
        target = tmp_path / go_mod.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(go_mod, target)
    return tmp_path


def _edit(root: Path, rel: str, old: str, new: str) -> None:
    """Replace the one occurrence of ``old`` in a copied file.

    Parameters
    ----------
    root : Path
        The copy's root.
    rel : str
        The file, relative to the root.
    old : str
        Text that occurs exactly once.
    new : str
        Its replacement.
    """
    path = root / rel
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, (rel, old)
    path.write_text(text.replace(old, new), encoding="utf-8")


def _problems(root: Path) -> list[str]:
    """Return the check's problems for a copy."""
    return check(root, load_manifest(root / MANIFEST))


def test_the_repository_agrees_with_its_manifest() -> None:
    """Every declaration of the checkout gives the manifest's answer."""
    assert check(ROOT, load_manifest(ROOT / MANIFEST)) == []


def test_the_scan_finds_every_kind_of_declaration() -> None:
    """The scan reaches runners, setup versions, toolchains, packages and the other files."""
    kinds = {declaration.tool for declaration in collect(ROOT)}
    expected = {"runner", "python", "go", "node", "julia", "pixi", "rust", "rust-stable", "mojo"}
    expected |= {"npm", "npm-sha256", "hdl-sdk", "apt:iverilog", "apt:cvc5", "choco:iverilog"}
    assert expected <= kinds


@pytest.mark.parametrize(
    ("rel", "old", "new", "named"),
    [
        (
            ".github/workflows/stale.yml",
            "runs-on: ubuntu-24.04",
            "runs-on: ubuntu-latest",
            "ubuntu-latest",
        ),
        (
            ".github/workflows/stale.yml",
            "runs-on: ubuntu-24.04",
            "runs-on: ubuntu-26.04",
            "ubuntu-26.04",
        ),
        (
            ".github/workflows/v3-engine.yml",
            "matrix.os == 'ubuntu-24.04'",
            "matrix.os == 'macos-latest'",
            "macos-latest",
        ),
        (
            ".github/workflows/fuzz.yml",
            "toolchain: nightly-2026-11-13",
            "toolchain: nightly",
            "'nightly'",
        ),
        (
            ".github/workflows/fuzz.yml",
            "cargo +nightly-2026-11-13 fuzz",
            "cargo +nightly fuzz",
            "'nightly'",
        ),
        (
            ".github/workflows/v3-engine.yml",
            "iverilog=12.0-2build2",
            "iverilog",
            "'iverilog'",
        ),
        (
            ".github/workflows/yosys-synth.yml",
            "cvc5=1.1.2-1build1",
            "cvc5=1.1.3-1",
            "'cvc5=1.1.3-1'",
        ),
        (
            ".github/workflows/v3-engine.yml",
            "iverilog --version=11.0",
            "iverilog --version=12.0",
            "'12.0'",
        ),
        (".github/workflows/ci.yml", 'version: "1.13.3"', 'version: "1.13"', "julia '1.13'"),
        (".github/workflows/ci.yml", "openmpi-bin", "openmpi-bin iverilog", "'iverilog'"),
        (
            ".github/workflows/stale.yml",
            "    steps:\n",
            "    steps:\n      - uses: actions/setup-java@v5\n",
            "setup-java",
        ),
        (
            ".github/workflows/hdl-toolchain.yml",
            "SCN_HDL_SDK_SHA256: 49a6",
            "SCN_HDL_SDK_SHA256: 59a6",
            "hdl-sdk",
        ),
        (
            ".github/actions/setup-npm/action.yml",
            "npm/-/npm-12.2.1.tgz",
            "npm/-/npm-12.2.0.tgz",
            "npm '12.2.0'",
        ),
        (
            "engine/rust-toolchain.toml",
            'channel = "1.98.1"',
            'channel = "1.99.1"',
            "rust-stable '1.99.1'",
        ),
        (
            "src/sc_neurocore/accel/mojo/pixi.toml",
            'mojo = "==1.0.0"',
            'mojo = "==1.2.0"',
            "mojo '==1.2.0'",
        ),
        ("requirements-dev.txt", "mojo==1.0.0", "mojo==1.2.0", "requirements-dev.txt"),
        ("src/sc_neurocore/accel/go/go.mod", "go 1.27.4", "go 1.27.3", "go '1.27.3'"),
    ],
)
def test_each_kind_of_drift_is_named(repo: Path, rel: str, old: str, new: str, named: str) -> None:
    """A declaration that leaves the manifest makes the check fail and name it.

    Parameters
    ----------
    repo : Path
        A copy of the files the check reads.
    rel : str
        The file to change.
    old : str
        The agreeing text.
    new : str
        The drifted text.
    named : str
        Text the problem must contain.
    """
    assert _problems(repo) == []
    _edit(repo, rel, old, new)
    problems = _problems(repo)
    assert problems and any(named in problem for problem in problems), problems


def test_a_choco_tool_outside_the_manifest_is_refused(repo: Path) -> None:
    """A Windows tool installed through choco needs its own manifest answer."""
    _edit(
        repo, ".github/workflows/v3-engine.yml", "choco install iverilog", "choco install verilator"
    )
    assert any("choco install verilator is not in the manifest" in p for p in _problems(repo))


def test_a_python_minor_is_accepted_only_in_a_wheel_job(repo: Path) -> None:
    """The minor alone passes in the listed wheel jobs and fails elsewhere."""
    _edit(
        repo,
        ".github/workflows/stale.yml",
        "    steps:\n",
        (
            '    steps:\n      - uses: actions/setup-python@v6\n        with:\n          python-version: "3.12"\n'
        ),
    )
    assert any("stale.yml:stale: python '3.12'" in p for p in _problems(repo))


STALE = ".github/workflows/stale.yml"
STEPS = "    steps:\n"


def test_a_matrix_reference_resolves_by_the_key_it_names(repo: Path) -> None:
    """A matrix key named other than the ``with`` key still gives its values to the check."""
    _edit(
        repo,
        STALE,
        "    runs-on: ubuntu-24.04\n" + STEPS,
        (
            "    runs-on: ubuntu-24.04\n    strategy:\n      matrix:\n        python: ['3.12.15']\n"
            + STEPS
            + "      - uses: actions/setup-python@v6\n        with:\n"
            + "          python-version: ${{ matrix.python }}\n"
        ),
    )
    assert any("stale.yml:stale: python '3.12.15'" in p for p in _problems(repo))


@pytest.mark.parametrize(
    "expression", ["${{ env.PY }}", "${{ vars.PY }}", "${{ inputs.python }}", "${{ matrix.none }}"]
)
def test_a_setup_version_without_a_literal_is_refused(repo: Path, expression: str) -> None:
    """A setup value the check cannot read as a literal is refused, not passed over.

    Parameters
    ----------
    repo : Path
        A copy of the files the check reads.
    expression : str
        A value that names no version literal.
    """
    _edit(
        repo,
        STALE,
        STEPS,
        STEPS
        + f"      - uses: actions/setup-python@v6\n        with:\n          python-version: {expression}\n",
    )
    assert any(f"python '{expression}'" in p for p in _problems(repo))


def test_a_run_script_matrix_reference_without_a_value_is_refused(repo: Path) -> None:
    """``cargo +${{ matrix.<key> }}`` with no such key stays visible and is refused."""
    _edit(repo, STALE, STEPS, STEPS + "      - run: cargo +${{ matrix.toolchain }} fuzz run x\n")
    assert any("stale.yml:stale: rust '${{'" in p for p in _problems(repo))


@pytest.mark.parametrize(
    "install",
    [
        "sudo apt-get -y install iverilog",
        "sudo apt install -y iverilog",
        "sudo apt -q -y install iverilog",
        "sudo apt-get -o Dpkg::Use-Pty=0 install -y iverilog",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends iverilog",
    ],
)
def test_every_apt_spelling_of_an_unpinned_install_is_refused(repo: Path, install: str) -> None:
    """``apt`` and ``apt-get``, with options before or after the verb, are both read.

    Parameters
    ----------
    repo : Path
        A copy of the files the check reads.
    install : str
        One spelling of an install of a listed tool without its version.
    """
    _edit(repo, STALE, STEPS, STEPS + f"      - run: {install}\n")
    assert any("stale.yml:stale: apt install 'iverilog'" in p for p in _problems(repo))


def test_a_pinned_apt_install_in_another_spelling_is_accepted(repo: Path) -> None:
    """The pin rule reads the version whatever the spelling of the command."""
    _edit(repo, STALE, STEPS, STEPS + "      - run: sudo apt install -y iverilog=12.0-2build2\n")
    assert _problems(repo) == []


@pytest.mark.parametrize(
    ("spelling", "problem"),
    [
        ("iverilog --version 11.0", None),
        ('iverilog --version "11.0"', None),
        ("iverilog --version 12.0", "choco iverilog version '12.0'"),
    ],
)
def test_the_choco_version_is_read_in_either_spelling(
    repo: Path, spelling: str, problem: str | None
) -> None:
    """``--version=V`` and ``--version V`` give the same declaration.

    Parameters
    ----------
    repo : Path
        A copy of the files the check reads.
    spelling : str
        The package and its version option.
    problem : str or None
        Text of the expected problem; ``None`` when the install agrees.
    """
    _edit(repo, ".github/workflows/v3-engine.yml", "iverilog --version=11.0", spelling)
    problems = _problems(repo)
    if problem is None:
        assert problems == []
    else:
        assert any(problem in p for p in problems), problems


@pytest.mark.parametrize(
    ("step", "named"),
    [
        ("    - uses: actions/setup-java@v5\n", "setup-java"),
        (
            "    - uses: actions/setup-node@v6\n      with:\n        node-version: '25.0.0'\n",
            "node '25.0.0'",
        ),
    ],
)
def test_composite_action_setup_steps_are_checked(repo: Path, step: str, named: str) -> None:
    """A ``uses:`` step inside a composite action answers the manifest like a workflow step.

    Parameters
    ----------
    repo : Path
        A copy of the files the check reads.
    step : str
        The step added to the composite action.
    named : str
        Text the problem must contain.
    """
    _edit(repo, ".github/actions/setup-npm/action.yml", "  steps:\n", "  steps:\n" + step)
    assert any(named in p for p in _problems(repo))


def test_only_the_npm_action_holds_the_npm_digest(repo: Path) -> None:
    """A digest in another composite action is not read as the npm archive's digest."""
    other = repo / ".github" / "actions" / "verify-other" / "action.yml"
    other.parent.mkdir()
    other.write_text(
        "name: Verify another archive\ndescription: test\nruns:\n  using: composite\n  steps:\n"
        "    - shell: bash\n      run: echo '" + "0" * 64 + "  archive.tgz' | sha256sum -c\n",
        encoding="utf-8",
    )
    assert _problems(repo) == []


def test_a_go_module_cache_below_src_is_not_a_declaration(repo: Path) -> None:
    """The Go toolchain's own ``go.mod`` files in a hidden cache are passed over; a source one is not."""
    cached = (
        repo
        / "src"
        / "sc_neurocore"
        / "accel"
        / "go"
        / ".venv"
        / "go"
        / "pkg"
        / "mod"
        / "x"
        / "go.mod"
    )
    cached.parent.mkdir(parents=True)
    cached.write_text("module x\n\ngo 1.24\n", encoding="utf-8")
    assert _problems(repo) == []
    source = repo / "src" / "sc_neurocore" / "accel" / "go" / "tools" / "go.mod"
    source.parent.mkdir(parents=True)
    source.write_text("module tools\n\ngo 1.24\n", encoding="utf-8")
    assert any("accel/go/tools/go.mod: go '1.24'" in p for p in _problems(repo))


def test_an_unreadable_rust_toolchain_file_is_reported(repo: Path) -> None:
    """Two toolchain files in one directory are a problem, not a crash."""
    (repo / "engine" / "rust-toolchain").write_text("1.98.1\n", encoding="utf-8")
    assert any("rust-stable 'unreadable" in p for p in _problems(repo))


@pytest.mark.parametrize(
    ("old", "new", "refusal"),
    [
        ('schema = "sc-neurocore.toolchain-manifest.v1"', 'schema = "other"', "is not of schema"),
        ('nightly = "nightly-2026-11-13"', 'nightly = "nightly"', "dated nightly"),
        ('main = "3.12.16"', 'main = "3.12.15"', "python.main"),
        (
            '[node]\nversion = "26.13.0"',
            '[node]\nversion = ""',
            "version must be a non-empty string",
        ),
        (
            'labels = ["ubuntu-24.04"',
            'labels = [1, "ubuntu-24.04"',
            "labels must be a non-empty list",
        ),
        ("[choco]\n", "[choco]\nbroken = 1\n", "[choco] must map"),
        ("[julia]\n", "[julia_moved]\n", "[julia] is missing"),
    ],
)
def test_a_malformed_manifest_is_refused(repo: Path, old: str, new: str, refusal: str) -> None:
    """A manifest field of the wrong shape refuses the load with a named reason.

    Parameters
    ----------
    repo : Path
        A copy holding the manifest.
    old : str
        Valid manifest text.
    new : str
        Its malformed replacement.
    refusal : str
        Text the refusal must contain.
    """
    _edit(repo, MANIFEST, old, new)
    with pytest.raises(ManifestError, match=refusal.replace("[", r"\[").replace(".", r"\.")):
        load_manifest(repo / MANIFEST)


def test_a_missing_or_shapeless_host_tool_list_is_refused(repo: Path) -> None:
    """Without host tools, or with a host tool that is not a table, the load fails."""
    path = repo / MANIFEST
    text = path.read_text(encoding="utf-8")
    path.write_text(text.split("[[host_tool]]", 1)[0], encoding="utf-8")
    with pytest.raises(ManifestError, match="host_tool"):
        load_manifest(path)
    head = text.split("[[host_tool]]", 1)[0]
    path.write_text(
        head.replace("\n[runners]", "\nhost_tool = [1]\n\n[runners]", 1), encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="must be a table"):
        load_manifest(path)
    path.unlink()
    with pytest.raises(ManifestError, match="cannot be read"):
        load_manifest(path)


def test_the_command_line_reports_each_outcome(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` returns 0 or 1, ``--list`` prints, an unreadable manifest returns 2.

    Parameters
    ----------
    repo : Path
        A copy of the files the check reads.
    capsys : pytest.CaptureFixture
        Captured output.
    """
    assert main(["--check"], root=repo) == 0
    assert "every declaration agrees" in capsys.readouterr().out
    assert main(["--list"], root=repo) == 0
    assert "choco:iverilog" in capsys.readouterr().out
    _edit(repo, ".github/workflows/stale.yml", "runs-on: ubuntu-24.04", "runs-on: ubuntu-latest")
    assert main(["--check"], root=repo) == 1
    output = capsys.readouterr().out
    assert "stale.yml:stale: runner 'ubuntu-latest'" in output
    assert output.rstrip().endswith("problems")
    (repo / MANIFEST).write_text("schema = 1\n", encoding="utf-8")
    assert main(["--check"], root=repo) == 2
    with pytest.raises(SystemExit):
        main([], root=repo)


def test_the_module_runs_as_a_command() -> None:
    """``python -m tools.toolchain_manifest --check`` passes on the checkout."""
    result = subprocess.run(
        [sys.executable, "-m", "tools.toolchain_manifest", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "every declaration agrees" in result.stdout


def _tool(directory: Path, name: str, script: str) -> None:
    """Write an executable shell script.

    Parameters
    ----------
    directory : Path
        Where the script goes.
    name : str
        Its file name.
    script : str
        Its body after the shebang.
    """
    path = directory / name
    path.write_text(f"#!/bin/sh\n{script}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_the_host_check_runs_each_tool_and_compares_its_line(tmp_path: Path) -> None:
    """Real executables answer; a wrong version, a missing tool, an unrunnable and a slow one are named.

    Parameters
    ----------
    tmp_path : Path
        Fresh directory for the tools and an empty repository root.
    """
    manifest = load_manifest(ROOT / MANIFEST)
    tools = tmp_path / "bin"
    tools.mkdir()
    answers = {
        "python": "echo 'Python 3.12.16'",
        "rustc": "echo 'rustc 1.98.1 (48a229cea 2026-09-01)'",
        "go": "echo 'go version go1.27.4 linux/amd64'",
        "node": "echo v26.13.0",
        "npm": "echo 12.2.1",
        "julia": "echo 'julia version 1.13.3'",
        "mojo": "echo 'Mojo 1.0.0 (ed45d567)'",
    }
    for host_tool in manifest.host_tools:
        answers[host_tool.command[0]] = f"echo '{host_tool.lines[0]}'"
    for name, script in answers.items():
        _tool(tools, name, script)
    root = tmp_path / "root"
    root.mkdir()
    assert check_host(root, manifest, path=str(tools)) == []

    _tool(tools, "z3", "echo 'Z3 version 4.16.0 - 64 bit'")
    (tools / "sv2v").unlink()
    (tools / "cvc5").write_text("#!/nonexistent/interpreter\n", encoding="utf-8")
    _tool(tools, "npm", f"exec {shutil.which('sleep')} 5")
    problems = check_host(root, manifest, path=str(tools), timeout=1)
    assert any(p.startswith("z3:") and "4.16.0" in p for p in problems)
    assert "sv2v: sv2v is not on PATH" in problems
    assert any(
        p.startswith("cvc5:") and "could not be run (FileNotFoundError)" in p for p in problems
    )
    assert any(p.startswith("npm:") and "TimeoutExpired" in p for p in problems)
    assert len(problems) == 4


def test_shapeless_jobs_and_steps_are_passed_over(repo: Path) -> None:
    """A job or step that is not a mapping yields nothing and stops nothing.

    Parameters
    ----------
    repo : Path
        A copy of the files the check reads.
    """
    data = {
        "jobs": {"a": "text", "b": {"runs-on": "ubuntu-24.04", "steps": ["text", {"run": "true"}]}}
    }
    found = list(workflow_declarations("w.yml", data))
    assert [(d.source, d.tool, d.value) for d in found] == [("w.yml:b", "runner", "ubuntu-24.04")]
    _edit(
        repo, ".github/actions/setup-npm/action.yml", "  steps:\n", "  steps:\n    - plain text\n"
    )
    assert _problems(repo) == []


def test_the_command_line_runs_the_host_check(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check-host`` reads PATH from the environment and fails on a tool that answers wrongly.

    Parameters
    ----------
    repo : Path
        A copy holding the manifest; it has no ``.venv``.
    tmp_path : Path
        Fresh directory for the tools.
    monkeypatch : pytest.MonkeyPatch
        Sets PATH to the tools only.
    capsys : pytest.CaptureFixture
        Captured output.
    """
    tools = tmp_path / "fake-bin"
    tools.mkdir()
    _tool(tools, "python", "echo 'Python 3.12.15'")
    monkeypatch.setenv("PATH", str(tools))
    assert main(["--check-host"], root=repo) == 1
    output = capsys.readouterr().out
    assert "python: " in output and "'Python 3.12.15'" in output
    assert "rustc: rustc is not on PATH" in output
