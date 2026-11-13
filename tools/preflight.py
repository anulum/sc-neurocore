#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Pre-push preflight gate — mirrors CI checks locally.

By default the command runs the gates that guard the files a change touched,
and no whole test suite: a gate whose kinds of file the change did not touch
prints that it is not applicable and does not run. ``--all-gates`` runs every
gate whatever was touched; ``--full-suite`` adds the whole pytest suite with
coverage. The frontend gate runs only for a change under ``studio/frontend/``.
When the touched files cannot be determined, every gate runs.
"""

import argparse
import fnmatch
import importlib
import pathlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from typing import cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SPDX_DIRS = ["src", "tests", "engine/src", "engine/tests", "engine/benches", "hdl", "bridge"]
SPDX_EXTS = {".py", ".rs", ".v"}
SPDX_MARKER = "SPDX-License-Identifier"

ENGINE_DIR = pathlib.Path("engine")
BENCHMARK_EVIDENCE_REPORT = (
    pathlib.Path(tempfile.gettempdir()) / "sc_neurocore_benchmark_evidence_gate_report.json"
)


def _coverage_fail_under() -> int:
    """Read the configured coverage floor, retaining 100 when configuration is unavailable."""
    pyproject = pathlib.Path("pyproject.toml")
    if not pyproject.exists():
        return 100
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return 100
    return int(data.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under", 100))


GATES = [
    ("cargo-fmt", ["cargo", "fmt", "--check", "--manifest-path", "engine/Cargo.toml"]),
    (
        "cargo-clippy",
        [
            "cargo",
            "clippy",
            "--all-targets",
            "--manifest-path",
            "engine/Cargo.toml",
            "--",
            "-D",
            "warnings",
        ],
    ),
    (
        "rust-doc-ratchet",
        ["python", "-m", "tools.rust_doc_ratchet"],
    ),
    (
        "cargo-rustdoc-symbols-fmt",
        ["cargo", "fmt", "--check", "--manifest-path", "tools/rustdoc_symbols/Cargo.toml"],
    ),
    (
        "cargo-rustdoc-symbols-clippy",
        [
            "cargo",
            "clippy",
            "--locked",
            "--all-targets",
            "--manifest-path",
            "tools/rustdoc_symbols/Cargo.toml",
            "--",
            "-D",
            "warnings",
        ],
    ),
    ("capability-manifest", ["python", "tools/capability_manifest.py", "--check"]),
    (
        "ruff-check",
        [
            "python",
            "-m",
            "ruff",
            "check",
            "src/",
            "tests/",
            "benchmarks/",
            "tools/capability_manifest.py",
            "tools/benchmark_evidence_gate.py",
        ],
    ),
    (
        "ruff-format",
        [
            "python",
            "-m",
            "ruff",
            "format",
            "--check",
            "src/",
            "tests/",
            "benchmarks/",
            "tools/capability_manifest.py",
            "tools/benchmark_evidence_gate.py",
        ],
    ),
    (
        "benchmark-evidence",
        [
            "python",
            "tools/benchmark_evidence_gate.py",
            "--manifest",
            "benchmarks/benchmark_regression_gates.json",
            "--output",
            str(BENCHMARK_EVIDENCE_REPORT),
        ],
    ),
    ("bandit", ["python", "-m", "bandit", "-r", "src/sc_neurocore/", "-c", "pyproject.toml", "-q"]),
    ("mypy", ["python", "-m", "mypy", "--strict", "src/sc_neurocore/"]),
    (
        "docstring-policy",
        ["python", "-m", "pytest", "tests/test_public_docstring_policy.py", "-q"],
    ),
    ("frontend", None),
    ("spdx-guard", None),
    (
        "pytest",
        None,
    ),
]

#: The pattern of a gate that guards every file: a generated file, a header or a
#: digest-bound record can go stale whatever changed, so such a gate always runs.
ANY_FILE = "*"
#: The tool's own path. A change of it changes how every gate is chosen and run,
#: so every gate applies to it.
PREFLIGHT_PATH = "tools/preflight.py"
_RUST_ENGINE_INPUTS = (
    "engine/*",
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/*",
    "bridge/vendor/*",
    "rustfmt.toml",
    ".rustfmt.toml",
    "clippy.toml",
    ".clippy.toml",
)
_RUSTDOC_SYMBOLS_INPUTS = (
    "tools/rustdoc_symbols/*",
    ".cargo/*",
    "bridge/vendor/*",
    "rustfmt.toml",
    ".rustfmt.toml",
    "clippy.toml",
    ".clippy.toml",
)
_RUFF_INPUTS = (
    "src/*",
    "tests/*",
    "benchmarks/*",
    "tools/capability_manifest.py",
    "tools/benchmark_evidence_gate.py",
    "pyproject.toml",
    "ruff.toml",
    ".ruff.toml",
)
#: The kinds of file each gate guards, as ``fnmatch`` patterns over
#: repository-relative POSIX paths (``*`` also matches ``/``). The Rust gates
#: guard their crate and every file that configures its build: the workspace
#: manifest and lock, the Cargo configuration and the vendored crate it builds
#: in place of the registry copy. The benchmark evidence gate binds the digests
#: of more than a thousand tracked files across the tree, so it guards any file,
#: as the generated-file and header gates do. A gate missing from this table
#: guards any file, so a gate added later is never skipped by omission.
GATE_GUARDS: dict[str, tuple[str, ...]] = {
    "cargo-fmt": _RUST_ENGINE_INPUTS,
    "cargo-clippy": _RUST_ENGINE_INPUTS,
    "rust-doc-ratchet": (
        *_RUST_ENGINE_INPUTS,
        "tools/rust_doc_*",
        "tools/doc_debt_ceiling.py",
        "tools/rustdoc_symbols/*",
    ),
    "cargo-rustdoc-symbols-fmt": _RUSTDOC_SYMBOLS_INPUTS,
    "cargo-rustdoc-symbols-clippy": _RUSTDOC_SYMBOLS_INPUTS,
    "capability-manifest": (ANY_FILE,),
    "ruff-check": _RUFF_INPUTS,
    "ruff-format": _RUFF_INPUTS,
    "benchmark-evidence": (ANY_FILE,),
    "bandit": ("src/sc_neurocore/*", "pyproject.toml"),
    "mypy": ("src/sc_neurocore/*", "pyproject.toml"),
    "docstring-policy": ("*.py", "*.pyi", "docs/docstring_policy.toml", "pyproject.toml"),
    "frontend": ("studio/frontend/*",),
    "spdx-guard": (ANY_FILE,),
    "pytest": (ANY_FILE,),
}
FRONTEND_DIR = pathlib.Path("studio/frontend")
#: The frontend's own scripts, run in :data:`FRONTEND_DIR` in this order: lint,
#: the type check of every project reference, then the tests (see
#: :func:`frontend_test_command`).
FRONTEND_CHECKS = (["npm", "run", "lint"], ["npx", "tsc", "-b"])
#: Printed by a run without ``--full-suite``.
NO_SUITE_RULE = (
    "The whole test suite is not run: a local preflight checks what the change touched, "
    "--full-suite adds the suite, and the hosted CI runs it on every push."
)


SPDX_SKIP_PARTS = {
    "__pycache__",
    ".pixi",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "build",
    "dist",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}


#: Files whose change alters what a pre-commit hook checks in every file, so the
#: range check must then cover all files, as the hosted Pre-commit run does.
#: typos reads ``_typos.toml`` and ``pyproject.toml`` by discovery beside the
#: ``.typos.toml`` its hook names.
HOOK_CONFIGURATION_PATHS = frozenset(
    {".pre-commit-config.yaml", ".typos.toml", "_typos.toml", "pyproject.toml"}
)
#: Branch a push is checked against when the checkout has no upstream configured.
DEFAULT_PUSH_TARGET = "origin/main"
#: Variable through which hooks that compare with a baseline commit, such as the
#: documentation scope guard and the Go and Rust documentation ratchets, receive
#: the push base; ``tools.docstring_policy_git.BASELINE_VARIABLE`` reads it.
POLICY_BASELINE_VARIABLE = "SC_NEUROCORE_POLICY_BASELINE"


def _git(*arguments: str) -> str | None:
    """Return the stripped output of a successful Git read, or None on any failure."""
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def push_base() -> str | None:
    """Return the commit a push of HEAD starts from, as the hosted push event records it.

    Returns
    -------
    str or None
        The merge base of HEAD with its upstream branch, or with
        :data:`DEFAULT_PUSH_TARGET` when no upstream is configured; None when
        Git cannot determine it.
    """
    target = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    return _git("merge-base", "HEAD", target or DEFAULT_PUSH_TARGET) or None


def touched_paths() -> list[str] | None:
    """Return the files a change touched: everything that differs from the push base.

    Returns
    -------
    list of str or None
        Sorted repository-relative paths that differ between the work tree and
        :func:`push_base` (committed, staged or not, a renamed file under both
        names), with the untracked files Git does not ignore. None when the base
        or either Git read cannot be determined, so that every gate runs.
    """
    base = push_base()
    if base is None:
        return None
    changed = _git("diff", "--name-only", "--no-renames", base)
    untracked = _git("ls-files", "--others", "--exclude-standard")
    if changed is None or untracked is None:
        return None
    return sorted({path for path in (*changed.splitlines(), *untracked.splitlines()) if path})


def gate_applies(name: str, touched: Sequence[str] | None) -> bool:
    """Return whether a gate guards any file the change touched.

    Parameters
    ----------
    name : str
        The gate, a key of :data:`GATE_GUARDS`; an undeclared gate guards any file.
    touched : sequence of str or None
        The paths from :func:`touched_paths`; None when they are unknown.

    Returns
    -------
    bool
        True when the touched files are unknown, when the gate guards any file,
        when the change touched :data:`PREFLIGHT_PATH`, or when a touched path
        matches one of the gate's patterns.
    """
    patterns = GATE_GUARDS.get(name, (ANY_FILE,))
    if touched is None or ANY_FILE in patterns or PREFLIGHT_PATH in touched:
        return True
    return any(fnmatch.fnmatchcase(path, pattern) for path in touched for pattern in patterns)


def frontend_files(touched: Sequence[str] | None) -> list[str] | None:
    """Return the absolute paths of the touched frontend files that still exist.

    Parameters
    ----------
    touched : sequence of str or None
        The paths from :func:`touched_paths`; None when they are unknown.

    Returns
    -------
    list of str or None
        None when the touched files are unknown, so the whole frontend test
        suite runs; otherwise the existing files under :data:`FRONTEND_DIR`, as
        absolute paths, so that no path can be read as an option.
    """
    if touched is None:
        return None
    prefix = f"{FRONTEND_DIR.as_posix()}/"
    return [
        str(pathlib.Path(path).resolve())
        for path in touched
        if path.startswith(prefix) and pathlib.Path(path).is_file()
    ]


def frontend_test_command(files: Sequence[str] | None) -> list[str] | None:
    """Return the frontend test command for the touched frontend files.

    Parameters
    ----------
    files : sequence of str or None
        The files from :func:`frontend_files`.

    Returns
    -------
    list of str or None
        ``vitest related --run`` over the files, which passes when no test
        relates to them; the whole ``vitest run`` when the files are unknown;
        None when no touched frontend file remains.
    """
    if files is None:
        return ["npx", "vitest", "run"]
    if not files:
        return None
    return ["npx", "vitest", "related", "--run", "--passWithNoTests", *files]


def run_frontend(files: Sequence[str] | None) -> bool:
    """Run the frontend's lint, type check and tests in its folder, stopping at a failure.

    Parameters
    ----------
    files : sequence of str or None
        The files from :func:`frontend_files`.

    Returns
    -------
    bool
        Whether every step passed. Without ``npm`` or ``npx`` on the search path
        the gate fails and says which: a frontend change cannot pass unchecked.
    """
    for tool in ("npm", "npx"):
        if shutil.which(tool) is None:
            print(f"  {tool} is not on the search path; a frontend change cannot pass unchecked")
            return False
    commands = [list(command) for command in FRONTEND_CHECKS]
    tests = frontend_test_command(files)
    if tests is None:
        print("  no touched frontend file remains, so no related test runs")
    else:
        commands.append(tests)
    return all(subprocess.run(command, cwd=FRONTEND_DIR).returncode == 0 for command in commands)


def range_gates(base: str, changed: list[str]) -> list[tuple[str, list[str]]]:
    """Build the gates that check the whole push range, as the hosted Pre-commit run does.

    Parameters
    ----------
    base : str
        Commit the push starts from, from :func:`push_base`.
    changed : list of str
        Paths changed between ``base`` and HEAD.

    Returns
    -------
    list of tuple
        The documentation scope guard with ``base`` as its baseline, and the
        pre-commit hooks over the files changed in the range. When a hook
        configuration file changed, the hooks run over all files instead. The
        hooks receive ``base`` through :data:`POLICY_BASELINE_VARIABLE`, so the
        hooks that compare with a baseline use the push base, as CI does.
    """
    if HOOK_CONFIGURATION_PATHS.intersection(changed):
        scope = ["--all-files"]
    else:
        scope = ["--from-ref", base, "--to-ref", "HEAD"]
    return [
        (
            "docstring-scope-range",
            ["python", "-m", "tools.docstring_policy_guard", "--since", base],
        ),
        (
            "pre-commit-range",
            [
                "env",
                f"{POLICY_BASELINE_VARIABLE}={base}",
                "python",
                "-m",
                "pre_commit",
                "run",
                "--show-diff-on-failure",
                *scope,
            ],
        ),
    ]


def check_spdx() -> bool:
    """Check maintained files beneath the working directory for direct SPDX headers.

    Returns
    -------
    bool
        Whether the shared repository header checker reports no missing paths.
        Missing paths are printed for repair; checker failures propagate.
    """
    try:
        module = importlib.import_module("spdx_header_audit")
    except ModuleNotFoundError:
        module = importlib.import_module("tools.spdx_header_audit")
    checker = cast(Callable[[pathlib.Path], list[str]], module.missing_direct_header_paths)
    missing = checker(pathlib.Path("."))
    if missing:
        print("Missing SPDX headers:")
        for f in missing:
            print(f"  {f}")
        return False
    return True


def _has_cargo() -> bool:
    """Require a real, successful Cargo version probe within ten seconds."""
    try:
        subprocess.run(["cargo", "--version"], capture_output=True, check=True, timeout=10)
        return True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def run_gate(name: str, cmd: list[str] | None) -> bool:
    """Execute one gate and report its actual success or prerequisite refusal.

    Parameters
    ----------
    name : str
        Gate label. Labels beginning with ``cargo-`` require a successful Cargo
        probe and an actual ``engine`` directory on every invocation.
    cmd : list[str] or None
        Command arguments for an external gate. ``spdx-guard`` and ``pytest``
        use their internal dispatch instead; other absent commands are refused.
        For ``frontend`` it holds the touched frontend files of
        :func:`frontend_files`, or None when they are unknown.

    Returns
    -------
    bool
        Whether the selected check succeeds. Missing Cargo prerequisites return
        False. Missing external commands and nonzero exit statuses also return False.
    """
    print(f"\n{'=' * 60}")
    print(f"  GATE: {name}")
    print(f"{'=' * 60}")
    if name.startswith("cargo-"):
        if not _has_cargo() or not ENGINE_DIR.is_dir():
            print(f"  FAIL: {name} (Cargo unavailable or engine/ is not a directory)")
            return False
    if name == "frontend":
        ok = run_frontend(cmd)
    elif name == "spdx-guard":
        ok = check_spdx()
    elif name == "pytest":
        ok = (
            subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/",
                    "-v",
                    "--cov=sc_neurocore",
                    "--cov-report=term",
                    f"--cov-fail-under={_coverage_fail_under()}",
                ]
            ).returncode
            == 0
        )
    elif cmd is not None:
        ok = subprocess.run(cmd).returncode == 0
    else:
        ok = False
    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def selected_gates(no_tests: bool, all_gates: bool = False) -> list[tuple[str, list[str] | None]]:
    """Return the gates one preflight run executes, with the push-range gates last.

    Parameters
    ----------
    no_tests : bool
        Leave out the pytest gate; every run without ``--full-suite`` does.
    all_gates : bool
        Run every configured gate whatever the change touched, as
        ``--all-gates`` requests. The frontend gate still needs a touched
        frontend file, or an unknown touched set.

    Returns
    -------
    list of tuple
        The configured gates that apply to the touched files of
        :func:`touched_paths` (each other gate prints that it is not
        applicable), then the range gates of :func:`range_gates` for the
        commits a push of HEAD would add. The frontend gate carries the touched
        frontend files of :func:`frontend_files` in place of a command. The
        range hooks read the working tree, while a push sends only HEAD, so the
        checkout must be clean. When the push base cannot be determined, or the
        checkout is not clean or its state cannot be read, a ``push-range`` gate
        without a command, which fails, takes their place.
    """
    touched = touched_paths()
    gates: list[tuple[str, list[str] | None]] = []
    for name, command in GATES:
        if no_tests and name == "pytest":
            continue
        if (name == "frontend" or not all_gates) and not gate_applies(name, touched):
            print(f"not applicable to this change: {name}")
            continue
        gates.append((name, frontend_files(touched) if name == "frontend" else command))
    base = push_base()
    status = _git("status", "--porcelain")
    if base is None:
        print("The push base could not be determined; the range gates refuse.")
        gates.append(("push-range", None))
    elif status is None:
        print("The checkout state could not be read; the range gates refuse.")
        gates.append(("push-range", None))
    elif status:
        print("The checkout has changes a push would not send; the range gates refuse:")
        print(status)
        gates.append(("push-range", None))
    else:
        changed = (_git("diff", "--name-only", base, "HEAD") or "").splitlines()
        gates.extend(range_gates(base, changed))
    return gates


def main() -> int:
    """Run the requested preflight gates and return one if any gate fails.

    By default the command runs the gates that guard the touched files and no
    test suite, and prints :data:`NO_SUITE_RULE`. ``--all-gates`` runs every
    gate whatever was touched (a full local check); ``--full-suite`` adds
    the whole pytest suite with coverage. ``--no-tests`` is accepted as the
    default, and ``--coverage`` is accepted and changes nothing. The range
    gates from :func:`range_gates` always run last, over the commits a push of
    HEAD would add.
    """
    parser = argparse.ArgumentParser(description="SC-NeuroCore preflight checks")
    suite = parser.add_mutually_exclusive_group()
    suite.add_argument(
        "--full-suite",
        action="store_true",
        help="Also run the whole pytest suite with coverage (not run by default)",
    )
    suite.add_argument(
        "--no-tests", action="store_true", help="The default: no test suite (kept for callers)"
    )
    parser.add_argument(
        "--all-gates",
        action="store_true",
        help="Run every gate whatever the change touched (a full local check)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Accepted and changes nothing: the suite of --full-suite always measures coverage",
    )
    args = parser.parse_args()

    if not args.full_suite:
        print(NO_SUITE_RULE)
    gates = selected_gates(no_tests=not args.full_suite, all_gates=args.all_gates)
    passed, failed = [], []
    for name, cmd in gates:
        if run_gate(name, cmd):
            passed.append(name)
        else:
            failed.append(name)

    print(f"\n{'=' * 60}")
    print(f"  PREFLIGHT: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
