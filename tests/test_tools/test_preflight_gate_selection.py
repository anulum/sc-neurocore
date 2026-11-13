# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — which preflight gates a change selects, observed through recording stand-ins

"""Exercise the preflight's gate selection on real Git repositories built by each test.

The external tools a gate starts (``python``, ``cargo``, ``npm`` and ``npx``) are
replaced, on the search path of the run, by stand-ins that only record their
command line and exit with 0, or with 1 when the test names them. They observe
which commands the preflight selects; they do not stand for what the tools would
report. Git and ``env`` are the real programs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import tools.preflight
from tools.preflight import ANY_FILE, GATE_GUARDS, GATES, gate_applies, main, touched_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
_STAND_INS = ("python", "cargo", "npm", "npx")
_STAND_IN = """#!/bin/sh
name=${0##*/}
{ printf '%s' "$name"; for argument in "$@"; do printf ' %s' "$argument"; done; printf '\\n'; } >> "$PREFLIGHT_RECORD"
case " $PREFLIGHT_FAIL " in *" $name "*) exit 1 ;; esac
exit 0
"""
_BASE_FILES = (
    "README.md",
    "engine/src/lib.rs",
    "src/sc_neurocore/model.py",
    "studio/frontend/src/view.ts",
    "studio/frontend/src/view.test.ts",
)
_RULE = "The whole test suite is not run"


def _git(root: Path, *arguments: str) -> str:
    """Run a real Git command in a fixture repository and return its output."""
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Preflight Fixture",
            "-c",
            "user.email=preflight@example.invalid",
            *arguments,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


def _write(root: Path, relative: str, text: str = "content\n") -> None:
    """Write one file of a fixture repository, creating its folders."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit_all(root: Path, message: str) -> None:
    """Stage every change of a fixture repository and commit it."""
    _git(root, "add", "--all")
    _git(root, "commit", "--quiet", "--message", message)


def _change(root: Path, *relatives: str) -> None:
    """Change the named files of a checkout and commit them, so the checkout stays clean."""
    for relative in relatives:
        _write(root, relative, f"changed {relative}\n")
    _commit_all(root, "Change")


def _clone_with_origin(tmp_path: Path) -> Path:
    """Create a bare origin holding the base files on main and a clone that tracks it."""
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--quiet", "--initial-branch=main")
    for relative in _BASE_FILES:
        _write(seed, relative)
    (seed / "tools").mkdir()
    for name in ("preflight.py", "spdx_header_audit.py"):
        shutil.copy(REPO_ROOT / "tools" / name, seed / "tools" / name)
    _commit_all(seed, "Base")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "--quiet", "--bare", str(seed), str(origin))
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "--quiet", str(origin), str(clone))
    return clone


def _lone_repository(tmp_path: Path) -> Path:
    """Create a repository with the base files and nothing to push to."""
    lone = tmp_path / "lone"
    lone.mkdir()
    _git(lone, "init", "--quiet", "--initial-branch=main")
    for relative in _BASE_FILES:
        _write(lone, relative)
    _commit_all(lone, "Base")
    return lone


def _search_path(tmp_path: Path, stand_ins: tuple[str, ...]) -> str:
    """Return a search path of recording stand-ins plus the real Git and env, and nothing else."""
    stand_in_folder = tmp_path / ("stand-ins-" + "-".join(stand_ins))
    stand_in_folder.mkdir(exist_ok=True)
    for name in stand_ins:
        path = stand_in_folder / name
        path.write_text(_STAND_IN, encoding="utf-8")
        path.chmod(0o755)
    real_folder = tmp_path / "real-tools"
    real_folder.mkdir(exist_ok=True)
    for name in ("git", "env"):
        found = shutil.which(name)
        assert found is not None, f"the real {name} is needed to build the fixture"
        link = real_folder / name
        if not link.is_symlink():
            link.symlink_to(found)
    return f"{stand_in_folder}{os.pathsep}{real_folder}"


class Run:
    """The status, output and recorded commands of one preflight run."""

    def __init__(self, status: int, output: str, record: Path) -> None:
        """Keep one run's results.

        Parameters
        ----------
        status : int
            The run's exit status.
        output : str
            What the run printed.
        record : Path
            The file the stand-ins appended their command lines to.
        """
        self.status = status
        self.output = output
        self.commands = record.read_text(encoding="utf-8").splitlines() if record.exists() else []

    def started(self, prefix: str) -> bool:
        """Return whether a recorded command line begins with ``prefix``.

        Parameters
        ----------
        prefix : str
            The start of a command line, such as ``"cargo fmt"``.

        Returns
        -------
        bool
            Whether any stand-in recorded such a command.
        """
        return any(command.startswith(prefix) for command in self.commands)

    def frontend(self) -> list[str]:
        """Return the recorded ``npm`` and ``npx`` command lines, in order.

        Returns
        -------
        list of str
            The frontend gate's commands as the stand-ins recorded them.
        """
        return [command for command in self.commands if command.split()[0] in {"npm", "npx"}]


Preflight = Callable[..., Run]


@pytest.fixture
def preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Preflight:
    """Return a function that runs the preflight in process in a checkout, under stand-ins.

    Parameters
    ----------
    tmp_path : Path
        Where the stand-ins and the command records are kept.
    monkeypatch : pytest.MonkeyPatch
        Sets the working folder, the search path and the arguments of each run.
    capsys : pytest.CaptureFixture
        Captures what each run prints.

    Returns
    -------
    callable
        ``run(checkout, *flags, stand_ins=..., fail="")`` returning a :class:`Run`.
    """

    def run(
        checkout: Path, *flags: str, stand_ins: tuple[str, ...] = _STAND_INS, fail: str = ""
    ) -> Run:
        record = tmp_path / f"record-{len(list(tmp_path.glob('record-*')))}.log"
        monkeypatch.chdir(checkout)
        monkeypatch.setenv("PATH", _search_path(tmp_path, stand_ins))
        monkeypatch.setenv("PREFLIGHT_RECORD", str(record))
        monkeypatch.setenv("PREFLIGHT_FAIL", fail)
        monkeypatch.setattr(sys, "argv", ["preflight.py", *flags])
        capsys.readouterr()
        status = main()
        return Run(status, capsys.readouterr().out, record)

    return run


# --- The frontend gate (design point a) ---


def test_a_change_without_a_frontend_file_does_not_call_npm(
    tmp_path: Path, preflight: Preflight
) -> None:
    """A Python change selects no frontend command."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "src/sc_neurocore/model.py")
    run = preflight(clone)
    assert run.frontend() == []
    assert "not applicable to this change: frontend" in run.output


def test_a_frontend_change_runs_lint_type_check_and_its_related_tests_in_order(
    tmp_path: Path, preflight: Preflight
) -> None:
    """Lint, then the type check, then the tests related to the touched file, by absolute path."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "studio/frontend/src/view.ts")
    run = preflight(clone)
    touched = clone.resolve() / "studio/frontend/src/view.ts"
    assert run.frontend() == [
        "npm run lint",
        "npx tsc -b",
        f"npx vitest related --run --passWithNoTests {touched}",
    ]
    assert "PASS: frontend" in run.output


def test_the_frontend_commands_run_in_the_frontend_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every frontend command starts in ``studio/frontend``, where its scripts are defined."""
    clone = _clone_with_origin(tmp_path)
    folder = tmp_path / "pwd-stand-in"
    folder.mkdir()
    record = tmp_path / "pwd.log"
    for name in ("npm", "npx"):
        path = folder / name
        path.write_text(f'#!/bin/sh\npwd >> "{record}"\n', encoding="utf-8")
        path.chmod(0o755)
    monkeypatch.chdir(clone)
    monkeypatch.setenv("PATH", f"{folder}{os.pathsep}{os.environ['PATH']}")
    assert tools.preflight.run_gate("frontend", [str(clone / "studio/frontend/src/view.ts")])
    folders = record.read_text(encoding="utf-8").split()
    assert folders == [str((clone / "studio/frontend").resolve())] * 3
    assert "PASS: frontend" in capsys.readouterr().out


def test_a_failing_frontend_step_stops_the_gate(tmp_path: Path, preflight: Preflight) -> None:
    """When lint fails, the type check and the tests do not start and the gate fails."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "studio/frontend/src/view.ts")
    run = preflight(clone, fail="npm")
    assert run.frontend() == ["npm run lint"]
    assert "FAIL: frontend" in run.output
    assert run.status == 1


def test_a_frontend_change_without_npm_fails_and_says_why(
    tmp_path: Path, preflight: Preflight
) -> None:
    """Without npm on the search path, a frontend change cannot pass."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "studio/frontend/src/view.ts")
    run = preflight(clone, stand_ins=("python", "cargo"))
    assert "npm is not on the search path" in run.output
    assert "FAIL: frontend" in run.output
    assert run.status == 1


def test_a_frontend_change_with_npm_but_without_npx_fails_and_says_why(
    tmp_path: Path, preflight: Preflight
) -> None:
    """The type check and the tests start through npx, so it is required as well."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "studio/frontend/src/view.ts")
    run = preflight(clone, stand_ins=("python", "cargo", "npm"))
    assert "npx is not on the search path" in run.output
    assert run.frontend() == []
    assert "FAIL: frontend" in run.output


def test_a_frontend_change_that_only_deletes_runs_lint_and_type_check(
    tmp_path: Path, preflight: Preflight
) -> None:
    """A deleted file has no related tests; lint and the type check still run."""
    clone = _clone_with_origin(tmp_path)
    (clone / "studio/frontend/src/view.ts").unlink()
    _commit_all(clone, "Delete")
    run = preflight(clone)
    assert run.frontend() == ["npm run lint", "npx tsc -b"]
    assert "no touched frontend file remains" in run.output


def test_an_unknown_change_runs_the_whole_frontend_test_suite(
    tmp_path: Path, preflight: Preflight
) -> None:
    """When the touched set cannot be determined, the frontend gate runs all its tests."""
    run = preflight(_lone_repository(tmp_path))
    assert run.frontend() == ["npm run lint", "npx tsc -b", "npx vitest run"]


# --- Gates that know what they guard (amendment, points e to g) ---


def test_a_markdown_change_starts_no_rust_or_python_analysis_but_runs_the_whole_tree_gates(
    tmp_path: Path, preflight: Preflight
) -> None:
    """One Markdown file: no cargo, mypy or bandit; the manifest, header and evidence gates run."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "README.md")
    run = preflight(clone)
    assert not run.started("cargo fmt")
    assert not run.started("cargo clippy")
    assert not run.started("python -m mypy")
    assert not run.started("python -m bandit")
    assert run.started("python tools/capability_manifest.py --check")
    assert run.started("python tools/benchmark_evidence_gate.py")
    assert "GATE: spdx-guard" in run.output
    for gate in ("cargo-fmt", "cargo-clippy", "mypy", "bandit", "ruff-check"):
        assert f"not applicable to this change: {gate}" in run.output


def test_a_rust_change_starts_the_rust_gates(tmp_path: Path, preflight: Preflight) -> None:
    """A change in the engine crate starts formatting, Clippy and the documentation ratchet."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "engine/src/lib.rs")
    run = preflight(clone)
    assert run.started("cargo fmt --check --manifest-path engine/Cargo.toml")
    assert run.started("cargo clippy --all-targets --manifest-path engine/Cargo.toml")
    assert run.started("python -m tools.rust_doc_ratchet")
    assert not run.started("python -m mypy")


def test_a_python_change_starts_the_python_gates(tmp_path: Path, preflight: Preflight) -> None:
    """A change in the package starts ruff, bandit, strict typing and the docstring policy."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "src/sc_neurocore/model.py")
    run = preflight(clone)
    for prefix in (
        "python -m ruff check",
        "python -m ruff format --check",
        "python -m bandit",
        "python -m mypy --strict src/sc_neurocore/",
        "python -m pytest tests/test_public_docstring_policy.py -q",
    ):
        assert run.started(prefix), prefix
    assert not run.started("cargo fmt")


def test_an_unknown_change_runs_every_gate(tmp_path: Path, preflight: Preflight) -> None:
    """Without a push base the touched set is unknown, so every gate runs (fail safe)."""
    run = preflight(_lone_repository(tmp_path))
    assert run.started("cargo fmt")
    assert run.started("python -m mypy")
    assert "not applicable to this change" not in run.output


def test_all_gates_runs_every_gate_whatever_was_touched(
    tmp_path: Path, preflight: Preflight
) -> None:
    """``--all-gates`` on a Markdown change still starts the Rust and Python gates."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "README.md")
    run = preflight(clone, "--all-gates")
    assert run.started("cargo fmt")
    assert run.started("cargo clippy")
    assert run.started("python -m mypy")
    assert run.started("python -m bandit")
    skipped = [line for line in run.output.splitlines() if line.startswith("not applicable")]
    assert skipped == ["not applicable to this change: frontend"]


def test_all_gates_still_runs_the_frontend_gate_only_for_a_frontend_change(
    tmp_path: Path, preflight: Preflight
) -> None:
    """The frontend gate needs a frontend change also under ``--all-gates`` (design point a)."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "README.md")
    assert preflight(clone, "--all-gates").frontend() == []
    _change(clone, "studio/frontend/src/view.ts")
    assert preflight(clone, "--all-gates").frontend()[:2] == ["npm run lint", "npx tsc -b"]


def test_a_change_of_the_preflight_itself_runs_every_gate(
    tmp_path: Path, preflight: Preflight
) -> None:
    """The tool that selects the gates decides for all of them, so its change runs them all."""
    clone = _clone_with_origin(tmp_path)
    _write(clone, "tools/preflight.py", (REPO_ROOT / "tools/preflight.py").read_text() + "\n")
    _commit_all(clone, "Change the preflight")
    run = preflight(clone)
    assert run.started("cargo fmt")
    assert run.started("python -m mypy")
    assert run.frontend()[:2] == ["npm run lint", "npx tsc -b"]


def test_the_whole_tree_gates_guard_any_file() -> None:
    """The generated-file, header and evidence gates declare any file (point g)."""
    for gate in ("capability-manifest", "spdx-guard", "benchmark-evidence"):
        assert GATE_GUARDS[gate] == (ANY_FILE,)
        assert gate_applies(gate, [])


def test_every_configured_gate_declares_what_it_guards() -> None:
    """A gate added later without a declaration would run always; the table names every gate."""
    assert {name for name, _ in GATES} == set(GATE_GUARDS)


def test_a_gate_without_a_declaration_always_applies() -> None:
    """Fail safe: an undeclared gate is never skipped by omission."""
    assert gate_applies("an-undeclared-gate", ["README.md"])
    assert gate_applies("an-undeclared-gate", [])


def test_a_declared_gate_applies_only_to_its_kinds_of_file() -> None:
    """``*`` in a pattern matches across folders; an unrelated path does not apply."""
    assert gate_applies("mypy", ["src/sc_neurocore/deep/folder/module.py"])
    assert not gate_applies("mypy", ["docs/page.md"])
    assert gate_applies("mypy", None)


def test_the_touched_set_holds_committed_staged_modified_and_untracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything that differs from the push base counts, renames by both names."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "committed.md")
    _git(clone, "mv", "README.md", "RENAMED.md")
    _write(clone, "src/sc_neurocore/model.py", "modified\n")
    _write(clone, "untracked.txt")
    monkeypatch.chdir(clone)
    assert touched_paths() == [
        "README.md",
        "RENAMED.md",
        "committed.md",
        "src/sc_neurocore/model.py",
        "untracked.txt",
    ]


def test_the_touched_set_is_unknown_without_a_base_or_when_git_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No push base, or a Git read that fails, gives no touched set."""
    monkeypatch.chdir(_lone_repository(tmp_path))
    assert touched_paths() is None
    clone = _clone_with_origin(tmp_path)
    monkeypatch.chdir(clone)
    index = clone / ".git" / "index"
    index.chmod(0)
    try:
        assert touched_paths() is None
    finally:
        index.chmod(0o644)


# --- No whole suite by default (design point b) ---


def test_the_default_starts_no_test_suite_and_names_the_rule(
    tmp_path: Path, preflight: Preflight
) -> None:
    """Without ``--full-suite`` the whole pytest suite does not start, and the run says why."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "src/sc_neurocore/model.py")
    run = preflight(clone)
    assert not run.started("python -m pytest tests/ ")
    assert _RULE in run.output


def test_full_suite_starts_the_whole_test_suite(tmp_path: Path, preflight: Preflight) -> None:
    """``--full-suite`` runs the suite with coverage, whatever the change."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "README.md")
    run = preflight(clone, "--full-suite")
    assert run.started("python -m pytest tests/ -v --cov=sc_neurocore")
    assert _RULE not in run.output


def test_no_tests_is_the_default(tmp_path: Path, preflight: Preflight) -> None:
    """``--no-tests`` selects exactly what the default selects."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "src/sc_neurocore/model.py")
    default = preflight(clone)
    no_tests = preflight(clone, "--no-tests")
    assert no_tests.commands == default.commands
    assert not no_tests.started("python -m pytest tests/ ")


def test_full_suite_and_no_tests_together_are_refused(tmp_path: Path, preflight: Preflight) -> None:
    """The two contradict each other, so the command refuses before any gate."""
    clone = _clone_with_origin(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        preflight(clone, "--full-suite", "--no-tests")
    assert stopped.value.code == 2


# --- The callers keep their behaviour (design point c, amendment point f) ---


def test_the_push_hook_runs_every_gate_and_no_suite() -> None:
    """The stored hook passes ``--all-gates`` and no ``--full-suite``."""
    hook = (REPO_ROOT / ".githooks/pre-push").read_text(encoding="utf-8")
    command = [line for line in hook.splitlines() if "tools/preflight.py" in line]
    assert command == ["python tools/preflight.py --all-gates"]


def test_the_make_targets_run_the_default_and_the_full_gate() -> None:
    """``make preflight`` runs the default; ``make preflight-full`` passes the flags."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipes: dict[str, list[str]] = {}
    target = ""
    for line in makefile.splitlines():
        if line and not line.startswith(("\t", "#", " ")) and ":" in line:
            target = line.split(":", 1)[0]
        elif line.startswith("\t"):
            recipes.setdefault(target, []).append(line.strip())
    assert recipes["preflight"] == ["python tools/preflight.py"]
    assert recipes["preflight-fast"] == ["python tools/preflight.py --no-tests"]
    assert recipes["preflight-full"] == ["python tools/preflight.py --full-suite --all-gates"]
    assert "preflight-full" in makefile.splitlines()[1]


# --- The command as a subprocess (entry point exercised) ---


def test_the_command_as_a_subprocess_selects_by_the_change(tmp_path: Path) -> None:
    """``python tools/preflight.py`` in a temporary clone, for a Markdown change."""
    clone = _clone_with_origin(tmp_path)
    _change(clone, "README.md")
    record = tmp_path / "subprocess.log"
    environment = {
        **os.environ,
        "PATH": _search_path(tmp_path, _STAND_INS),
        "PREFLIGHT_RECORD": str(record),
        "PREFLIGHT_FAIL": "",
    }
    done = subprocess.run(
        [sys.executable, "tools/preflight.py"],
        cwd=clone,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    run = Run(done.returncode, done.stdout, record)
    assert _RULE in run.output, done.stdout + done.stderr
    assert run.started("python tools/capability_manifest.py --check")
    assert not run.started("cargo")
    assert run.frontend() == []
    assert "not applicable to this change: mypy" in run.output
