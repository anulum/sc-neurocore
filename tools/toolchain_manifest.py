# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — One answer per tool, for the hosted runs and for this host

"""Hold every tool declaration of the repository to ``tools/toolchain_manifest.toml``.

``--check`` reads the workflows, the composite actions, the Rust toolchain
file, the ``go.mod`` files, the Mojo ``pixi.toml`` and ``requirements-dev.txt``.
It refuses a declaration that differs from
the manifest, a floating spelling (a ``*-latest`` runner, an undated
``nightly``, an install of a listed apt or choco tool without its version), a
setup version it cannot read as a literal (an ``env``, ``vars`` or ``inputs``
reference, or a matrix key with no value) and a setup action the manifest does
not know. ``--check-host`` asks each tool on
``PATH``, with the project's ``.venv/bin`` first, for its version. ``--list``
prints every declaration found and where. Run it as ``python -m
tools.toolchain_manifest`` from the repository root.

Exit status: 0 when everything agrees, 1 when a declaration or a host tool
differs, 2 when the manifest cannot be read.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
from typing import Any

import yaml

from tools.rust_toolchain_pin import RustToolchainPinError, pinned_toolchain

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "tools/toolchain_manifest.toml"
SCHEMA = "sc-neurocore.toolchain-manifest.v1"
FLOATING_RUNNER = re.compile(r"\b(?:ubuntu|windows|macos)-latest\b")
RUNNER_LABEL = re.compile(r"^(?:ubuntu|windows|macos)-")
CARGO_TOOLCHAIN = re.compile(r"\bcargo \+(\S+)")
QUOTED_VERSION = re.compile(r"'(\d+(?:\.\d+)+)'")
NPM_ARCHIVE = re.compile(r"npm/-/npm-(\d+\.\d+\.\d+)\.tgz")
SHA256 = re.compile(r"\b[0-9a-f]{64}\b")
CHOCO_INSTALL = re.compile(r"choco install (\S+)([^\n]*)")
#: ``apt-get install`` and ``apt install``, with options before the verb (``-y``, ``-o key=value``).
APT_INSTALL = re.compile(r"\bapt(?:-get)?(?:\s+(?:-o\s+\S+|-\S+))*\s+install\b((?:[^\n\\]|\\\n)*)")
MATRIX_REFERENCE = re.compile(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}")
#: Setup actions the manifest answers, by action path without its ref.
KNOWN_SETUP = {
    "actions/setup-python": "python",
    "actions/setup-go": "go",
    "actions/setup-node": "node",
    "julia-actions/setup-julia": "julia",
    "prefix-dev/setup-pixi": "pixi",
    "dtolnay/rust-toolchain": "rust",
    "./.github/actions/setup-npm": "npm",
}
#: Other declaration sites, relative to the repository root.
RUST_TOOLCHAIN_FILE = "engine/rust-toolchain.toml"
MOJO_PIXI = "src/sc_neurocore/accel/mojo/pixi.toml"
REQUIREMENTS_DEV = "requirements-dev.txt"
HDL_TOOLCHAIN = ".github/workflows/hdl-toolchain.yml"


class ManifestError(ValueError):
    """The manifest is missing, malformed or of another schema."""


@dataclass(frozen=True)
class HostTool:
    """One command whose output must hold an expected line on this host.

    Attributes
    ----------
    name : str
        The tool, as reported.
    command : tuple of str
        The command and its arguments.
    lines : tuple of str
        Accepted beginnings of one output line.
    """

    name: str
    command: tuple[str, ...]
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Manifest:
    """The one answer per tool that every declaration must give."""

    runners: frozenset[str]
    python: frozenset[str]
    python_main: str
    wheel_minors: frozenset[str]
    wheel_jobs: frozenset[str]
    rust_stable: str
    rust_nightly: str
    go: str
    node: str
    npm: str
    npm_sha256: str
    pixi: str
    julia: tuple[str, ...]
    mojo: str
    apt: Mapping[str, str]
    choco: Mapping[str, str]
    sdk_sha256: str
    host_tools: tuple[HostTool, ...]


def _strings(table: Mapping[str, Any], key: str) -> list[str]:
    """Return a manifest field that must be a non-empty list of strings."""
    value = table.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ManifestError(f"{key} must be a non-empty list of strings")
    return value


def _string(table: Mapping[str, Any], key: str) -> str:
    """Return a manifest field that must be a non-empty string."""
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{key} must be a non-empty string")
    return value


def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a manifest table."""
    value = data.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"[{key}] is missing")
    return value


def load_manifest(path: Path) -> Manifest:
    """Read and validate the manifest.

    Parameters
    ----------
    path : Path
        The TOML file.

    Returns
    -------
    Manifest
        Its answers.

    Raises
    ------
    ManifestError
        The file is unreadable, of another schema, or a field is missing or of the wrong type.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ManifestError(f"{path} cannot be read: {error}") from error
    if data.get("schema") != SCHEMA:
        raise ManifestError(f"{path} is not of schema {SCHEMA}")
    python, rust, julia, apt, choco = (
        _table(data, k) for k in ("python", "rust", "julia", "apt", "choco")
    )
    for name, table in (("apt", apt), ("choco", choco)):
        if not table or not all(isinstance(v, str) and v for v in table.values()):
            raise ManifestError(f"[{name}] must map each tool to a version string")
    tools = data.get("host_tool")
    if not isinstance(tools, list) or not tools:
        raise ManifestError("[[host_tool]] is missing")
    host_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ManifestError("each [[host_tool]] must be a table")
        host_tools.append(
            HostTool(
                _string(tool, "name"), tuple(_strings(tool, "command")), (_string(tool, "line"),)
            )
        )
    manifest = Manifest(
        runners=frozenset(_strings(_table(data, "runners"), "labels")),
        python=frozenset(_strings(python, "versions")),
        python_main=_string(python, "main"),
        wheel_minors=frozenset(_strings(python, "wheel_minors")),
        wheel_jobs=frozenset(_strings(python, "wheel_jobs")),
        rust_stable=_string(rust, "stable"),
        rust_nightly=_string(rust, "nightly"),
        go=_string(_table(data, "go"), "version"),
        node=_string(_table(data, "node"), "version"),
        npm=_string(_table(data, "npm"), "version"),
        npm_sha256=_string(_table(data, "npm"), "sha256"),
        pixi=_string(_table(data, "pixi"), "version"),
        julia=tuple(_strings(julia, "versions")),
        mojo=_string(_table(data, "mojo"), "version"),
        apt=dict(apt),
        choco=dict(choco),
        sdk_sha256=_string(_table(data, "hdl_sdk"), "archive_sha256"),
        host_tools=tuple(host_tools),
    )
    if manifest.python_main not in manifest.python:
        raise ManifestError("python.main must be one of python.versions")
    if not re.fullmatch(r"nightly-\d{4}-\d{2}-\d{2}", manifest.rust_nightly):
        raise ManifestError("rust.nightly must be a dated nightly, nightly-YYYY-MM-DD")
    return manifest


@dataclass(frozen=True)
class Declaration:
    """One tool version found in the repository.

    Attributes
    ----------
    source : str
        Repository-relative file and, for a workflow, the job.
    tool : str
        The manifest's name for the tool.
    value : str
        The version or label as written.
    """

    source: str
    tool: str
    value: str


def _matrix_values(job: Mapping[str, Any], key: str) -> list[str]:
    """Return every value a job's matrix gives ``key``, from its lists and its include entries."""
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    values: list[Any] = []
    listed = matrix.get(key)
    if isinstance(listed, list):
        values.extend(listed)
    for entry in matrix.get("include") or []:
        if isinstance(entry, dict) and key in entry:
            values.append(entry[key])
    return [str(v) for v in values]


def _nested_runner_values(job: Mapping[str, Any]) -> list[str]:
    """Return every string in a job's matrix that names a runner image."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and RUNNER_LABEL.match(node):
            found.append(node)

    walk((job.get("strategy") or {}).get("matrix") or {})
    return found


def _resolve(value: Any, job: Mapping[str, Any]) -> list[str]:
    """Return the literal versions a ``with`` value stands for.

    A plain value stands for itself, ``${{ matrix.<key> }}`` for the values the
    job's matrix gives that key, and another expression for the quoted version
    literals it holds. A value that resolves to no literal stands for its own
    text, so that it is refused rather than passed over.
    """
    text = str(value)
    if "${{" not in text:
        return [text]
    reference = MATRIX_REFERENCE.fullmatch(text)
    found = _matrix_values(job, reference.group(1)) if reference else QUOTED_VERSION.findall(text)
    return found or [text]


def workflow_declarations(rel: str, data: Mapping[str, Any]) -> Iterator[Declaration]:
    """Yield every tool declaration of one workflow file.

    Parameters
    ----------
    rel : str
        The file, relative to the repository root.
    data : Mapping
        The parsed workflow.

    Yields
    ------
    Declaration
        Each runner label, setup version and run-script toolchain or package found.
    """
    for job_name, job in (data.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        source = f"{rel}:{job_name}"
        runs_on = job.get("runs-on")
        if runs_on is not None and "${{" not in str(runs_on):
            yield Declaration(source, "runner", str(runs_on))
        for label in _nested_runner_values(job):
            yield Declaration(source, "runner", label)
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                yield from _step_declarations(source, job, step)


def _step_declarations(
    source: str, job: Mapping[str, Any], step: Mapping[str, Any]
) -> Iterator[Declaration]:
    """Yield the declarations of one step: its setup action and its run script."""
    uses = str(step.get("uses", "")).split("@", 1)[0]
    given = step.get("with") or {}
    keys = {"python": "python-version", "go": "go-version", "node": "node-version"}
    keys |= {"julia": "version", "pixi": "pixi-version", "rust": "toolchain"}
    tool = KNOWN_SETUP.get(uses)
    if tool in keys:
        for value in _resolve(given.get(keys[tool], ""), job):
            yield Declaration(source, tool, value)
    elif tool is None and uses and _is_setup_action(uses):
        yield Declaration(source, "unknown-setup", uses)
    script = MATRIX_REFERENCE.sub(
        lambda match: " ".join(_matrix_values(job, match.group(1))) or match.group(0),
        str(step.get("run", "") or ""),
    )
    yield from run_declarations(source, script)


def _is_setup_action(uses: str) -> bool:
    """Return whether an action installs a tool by its name."""
    last = uses.rstrip("/").rsplit("/", 1)[-1]
    return (last.startswith("setup-") and not uses.startswith("docker/")) or last.endswith(
        "-toolchain"
    )


def run_declarations(source: str, script: str) -> Iterator[Declaration]:
    """Yield the toolchains and packages a shell script names.

    Parameters
    ----------
    source : str
        Where the script stands.
    script : str
        The script's text.

    Yields
    ------
    Declaration
        Each ``cargo +toolchain``, apt package and choco install.
    """
    for toolchain in CARGO_TOOLCHAIN.findall(script):
        yield Declaration(source, "rust", toolchain.strip("\"'"))
    for words in APT_INSTALL.findall(script):
        for word in re.split(r"[\s\\]+", words):
            name = word.split("=", 1)[0]
            if word and not word.startswith("-"):
                yield Declaration(source, f"apt:{name}", word)
    for name, options in CHOCO_INSTALL.findall(script):
        version = re.search(r"--version[= ]\s*(\S+)", options)
        yield Declaration(source, f"choco:{name}", version.group(1).strip("\"'") if version else "")


def _declaration_problems(declaration: Declaration, manifest: Manifest) -> list[str]:
    """Return why one declaration differs from the manifest; empty when it agrees."""
    tool, value, where = declaration.tool, declaration.value, declaration.source
    job = where.split("/")[-1]
    answers: dict[str, Callable[[str], bool]] = {
        "runner": lambda v: v in manifest.runners,
        "python": lambda v: (
            v in manifest.python or (v in manifest.wheel_minors and job in manifest.wheel_jobs)
        ),
        "go": lambda v: v == manifest.go,
        "node": lambda v: v == manifest.node,
        "julia": lambda v: v in manifest.julia,
        "pixi": lambda v: v == manifest.pixi,
        "rust": lambda v: v in (manifest.rust_stable, manifest.rust_nightly),
    }
    if tool == "unknown-setup":
        return [f"{where}: setup action {value} installs a tool the manifest does not list"]
    if tool.startswith("apt:"):
        name = tool[4:]
        if name not in manifest.apt:
            return []
        expected = f"{name}={manifest.apt[name]}"
        return (
            []
            if value == expected
            else [f"{where}: apt install {value!r}, the manifest answers {expected!r}"]
        )
    if tool.startswith("choco:"):
        name = tool[6:]
        version = manifest.choco.get(name)
        if version is None:
            return [f"{where}: choco install {name} is not in the manifest"]
        return (
            []
            if value == version
            else [f"{where}: choco {name} version {value!r}, the manifest answers {version!r}"]
        )
    check = answers.get(tool)
    if check is None or check(value):
        return []
    return [f"{where}: {tool} {value!r} is not the manifest's answer"]


def _workflow_files(root: Path) -> list[Path]:
    """Return the workflow and composite action files, sorted."""
    github = root / ".github"
    return sorted(github.glob("workflows/*.yml")) + sorted(github.glob("actions/*/action.yml"))


def _go_mod_files(root: Path) -> list[Path]:
    """Return every ``go.mod`` below ``src`` outside hidden directories.

    A local Go module cache, such as ``.venv/go/pkg/mod``, holds the Go
    toolchain's own ``go.mod`` files; they are not this repository's
    declarations.
    """
    source = root / "src"
    return sorted(
        path
        for path in source.rglob("go.mod")
        if not any(part.startswith(".") for part in path.relative_to(source).parts[:-1])
    )


def collect(root: Path) -> list[Declaration]:
    """Return every declaration of every scanned file.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    list of Declaration
        In file order.
    """
    found: list[Declaration] = []
    for path in _workflow_files(root):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            for label in FLOATING_RUNNER.findall(line):
                found.append(Declaration(f"{rel}:{number}", "runner", label))
        data = yaml.safe_load(text) or {}
        if rel.startswith(".github/actions/"):
            for step in (data.get("runs") or {}).get("steps") or []:
                if isinstance(step, dict):
                    found.extend(_step_declarations(rel, {}, step))
            archives = NPM_ARCHIVE.findall(text)
            for version in archives:
                found.append(Declaration(rel, "npm", version))
            # Only the action that downloads the npm archive holds the npm digest.
            for digest in SHA256.findall(text) if archives else []:
                found.append(Declaration(rel, "npm-sha256", digest))
        else:
            found.extend(workflow_declarations(rel, data))
    try:
        pin = pinned_toolchain(root, "engine/Cargo.toml")
        channel = pin.channel if pin else ""
    except RustToolchainPinError as error:
        channel = f"unreadable ({error})"
    found.append(Declaration(RUST_TOOLCHAIN_FILE, "rust-stable", channel))
    for path in _go_mod_files(root):
        for version in re.findall(r"(?m)^go (\S+)$", path.read_text(encoding="utf-8")):
            found.append(Declaration(path.relative_to(root).as_posix(), "go", version))
    pixi = tomllib.loads((root / MOJO_PIXI).read_text(encoding="utf-8"))
    found.append(Declaration(MOJO_PIXI, "mojo", str(pixi.get("dependencies", {}).get("mojo", ""))))
    for pin_text in re.findall(
        r"(?m)^mojo==([^\s;]+)", (root / REQUIREMENTS_DEV).read_text(encoding="utf-8")
    ):
        found.append(Declaration(REQUIREMENTS_DEV, "mojo", f"=={pin_text}"))
    sdk = re.search(
        r"SCN_HDL_SDK_SHA256: ([0-9a-f]{64})", (root / HDL_TOOLCHAIN).read_text(encoding="utf-8")
    )
    found.append(Declaration(HDL_TOOLCHAIN, "hdl-sdk", sdk.group(1) if sdk else ""))
    return found


def check(root: Path, manifest: Manifest) -> list[str]:
    """Return every disagreement between the repository and the manifest.

    Parameters
    ----------
    root : Path
        The repository root.
    manifest : Manifest
        The answers.

    Returns
    -------
    list of str
        One sentence per problem, in file order; empty when all agree.
    """
    problems: list[str] = []
    fixed = {
        "rust-stable": manifest.rust_stable,
        "mojo": f"=={manifest.mojo}",
        "npm": manifest.npm,
        "npm-sha256": manifest.npm_sha256,
        "hdl-sdk": manifest.sdk_sha256,
    }
    for declaration in collect(root):
        if declaration.tool in fixed:
            if declaration.value != fixed[declaration.tool]:
                problems.append(
                    f"{declaration.source}: {declaration.tool} {declaration.value!r} "
                    f"is not the manifest's answer {fixed[declaration.tool]!r}"
                )
            continue
        problems.extend(_declaration_problems(declaration, manifest))
    return problems


def host_expectations(manifest: Manifest) -> list[HostTool]:
    """Return every host check: those derived from the versions, then the manifest's own.

    Parameters
    ----------
    manifest : Manifest
        The answers.

    Returns
    -------
    list of HostTool
        The command to run and the accepted beginnings of one output line.
    """
    derived = [
        HostTool("python", ("python", "--version"), (f"Python {manifest.python_main}",)),
        HostTool("rustc", ("rustc", "--version"), (f"rustc {manifest.rust_stable} ",)),
        HostTool("go", ("go", "version"), (f"go version go{manifest.go} ",)),
        HostTool("node", ("node", "--version"), (f"v{manifest.node}",)),
        HostTool("npm", ("npm", "--version"), (manifest.npm,)),
        HostTool(
            "julia", ("julia", "--version"), tuple(f"julia version {v}" for v in manifest.julia)
        ),
        HostTool("mojo", ("mojo", "--version"), (f"Mojo {manifest.mojo} ",)),
    ]
    return derived + list(manifest.host_tools)


def check_host(
    root: Path, manifest: Manifest, path: str | None = None, timeout: float = 60
) -> list[str]:
    """Return every host tool that is missing or answers another version.

    Parameters
    ----------
    root : Path
        The repository root; its ``.venv/bin`` comes first on the search path.
    manifest : Manifest
        The answers.
    path : str, optional
        The search path to use; defaults to the environment's ``PATH``.
    timeout : float
        Seconds each version command may take.

    Returns
    -------
    list of str
        One sentence per tool that does not answer; empty when all do.
    """
    search = os.pathsep.join(
        [str(root / ".venv" / "bin"), path if path is not None else os.environ.get("PATH", "")]
    )
    environment = {**os.environ, "PATH": search}
    problems: list[str] = []
    for tool in host_expectations(manifest):
        executable = shutil.which(tool.command[0], path=search)
        if executable is None:
            problems.append(f"{tool.name}: {tool.command[0]} is not on PATH")
            continue
        try:
            result = subprocess.run(
                [executable, *tool.command[1:]],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=environment,
                cwd=root,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            problems.append(f"{tool.name}: {executable} could not be run ({type(error).__name__})")
            continue
        output = (result.stdout + "\n" + result.stderr).splitlines()
        if not any(line.startswith(expected) for line in output for expected in tool.lines):
            first = output[0] if output else ""
            problems.append(
                f"{tool.name}: {executable} answers {first!r}, the manifest expects {tool.lines[0]!r}"
            )
    return problems


def main(argv: Sequence[str] | None = None, root: Path = ROOT) -> int:
    """Run the checks the arguments ask for.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.
    root : Path
        The repository root.

    Returns
    -------
    int
        0 when everything agrees, 1 on a disagreement, 2 when the manifest cannot be read.
    """
    parser = argparse.ArgumentParser(
        description="Hold every tool declaration to the toolchain manifest."
    )
    parser.add_argument(
        "--check", action="store_true", help="hold the repository's declarations to the manifest"
    )
    parser.add_argument(
        "--check-host", action="store_true", help="ask each tool on PATH for its version"
    )
    parser.add_argument("--list", action="store_true", help="print every declaration found")
    args = parser.parse_args(argv)
    if not (args.check or args.check_host or args.list):
        parser.error("choose --check, --check-host or --list")
    try:
        manifest = load_manifest(root / MANIFEST)
    except ManifestError as error:
        print(f"toolchain manifest: {error}", file=sys.stderr)
        return 2
    problems: list[str] = []
    if args.list:
        for declaration in collect(root):
            print(f"{declaration.tool:14} {declaration.value:32} {declaration.source}")
    if args.check:
        problems += check(root, manifest)
    if args.check_host:
        problems += check_host(root, manifest)
    for problem in problems:
        print(problem)
    if args.check or args.check_host:
        print(
            "toolchain manifest: "
            + (f"{len(problems)} problems" if problems else "every declaration agrees")
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
