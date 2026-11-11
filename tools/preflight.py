#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Pre-push preflight gate — mirrors CI checks locally."""

import argparse
import importlib
import pathlib
import subprocess
import sys
import tempfile
from collections.abc import Callable
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
    ("spdx-guard", None),
    (
        "pytest",
        None,
    ),
]


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
        configuration file changed, the hooks run over all files instead.
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
            ["python", "-m", "pre_commit", "run", "--show-diff-on-failure", *scope],
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
    if name == "spdx-guard":
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


def selected_gates(no_tests: bool) -> list[tuple[str, list[str] | None]]:
    """Return the gates one preflight run executes, with the push-range gates last.

    Parameters
    ----------
    no_tests : bool
        Leave out the pytest gate, as ``--no-tests`` requests.

    Returns
    -------
    list of tuple
        The configured gates, then the range gates of :func:`range_gates` for
        the commits a push of HEAD would add. The range hooks read the working
        tree, while a push sends only HEAD, so the checkout must be clean. When
        the push base cannot be determined, or the checkout is not clean or its
        state cannot be read, a ``push-range`` gate without a command, which
        fails, takes their place.
    """
    gates: list[tuple[str, list[str] | None]] = [
        (name, command) for name, command in GATES if not (no_tests and name == "pytest")
    ]
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

    The command runs all configured gates by default. ``--no-tests`` explicitly
    excludes the pytest gate; ``--coverage`` preserves the default coverage run.
    The range gates from :func:`range_gates` always run last, over the commits a
    push of HEAD would add. This entry point can execute the complete local suite.
    """
    parser = argparse.ArgumentParser(description="SC-NeuroCore preflight checks")
    parser.add_argument("--no-tests", action="store_true", help="Skip pytest (fast lint-only mode)")
    parser.add_argument(
        "--coverage", action="store_true", help="Same as default (tests always include coverage)"
    )
    args = parser.parse_args()

    gates = selected_gates(args.no_tests)
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
