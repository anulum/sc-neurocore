#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — compiled Go/Mojo accelerator backend builder

"""Build the compiled Go and Mojo accelerator shared libraries.

Each neuron backend is loaded through a ``ctypes`` loader that opens a
``lib<name>.so`` sitting **beside its own source file** (the Go source under
``accel/go/**`` and the Mojo source under ``accel/mojo/**``). The output name
is not always the source name — ``hindmarsh_rose.go`` builds ``libhr.so`` — so
the authoritative source→output pairing is taken from the ``go build`` and
``mojo build`` recipes embedded in the model modules. Parsing those recipes
means this builder cannot drift from what the loaders document.

The shared libraries are gitignored build artefacts, so CI must run this before
the backend/benchmark tests, otherwise every compiled-backend test fails with
``RuntimeError: lib<name>.so is not built``.

Usage::

    python tools/build_accel_backends.py --language all
    python tools/build_accel_backends.py --language go --require theta,adex
    python tools/build_accel_backends.py --language mojo \\
        --mojo-command "pixi run --manifest-path src/sc_neurocore/accel/mojo/pixi.toml mojo"
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ACCEL_ROOT = _REPO_ROOT / "src" / "sc_neurocore" / "accel"
_PRUNED_DIRS = frozenset({".pixi", "__pycache__", ".git", "node_modules", "target"})

# Authoritative build recipes embedded in the model / accel modules.
_GO_HINT = re.compile(r"buildmode=c-shared -o (lib[a-z0-9_]+\.so) ([a-z0-9_]+\.go)")
_MOJO_HINT = re.compile(r"emit shared-lib -o (lib[a-z0-9_]+\.so) ([a-z0-9_]+\.mojo)")


@dataclass(frozen=True)
class BackendTarget:
    """A single compiled backend: one source file, one output library."""

    language: str  # "go" or "mojo"
    name: str  # neuron/source stem, e.g. "hindmarsh_rose"
    source: Path  # absolute path to the .go / .mojo source
    output: Path  # absolute path of the lib<name>.so to produce (beside source)


@dataclass(frozen=True)
class BuildResult:
    """Outcome of building one target."""

    target: BackendTarget
    ok: bool
    detail: str


def _read_text(paths: Iterable[Path]) -> str:
    """Concatenate the text of every readable file in ``paths``."""
    chunks: list[str] = []
    for path in paths:
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(chunks)


def _hint_pairs(text: str, pattern: re.Pattern[str]) -> dict[str, str]:
    """Map source basename -> output library name for every recipe in ``text``."""
    pairs: dict[str, str] = {}
    for output, source in pattern.findall(text):
        pairs[source] = output
    return pairs


def _loader_lib_parts(node: ast.AST) -> tuple[str, list[str]] | None:
    """Return ``(root_name, segments)`` for a library path a loader constructs.

    Two idioms are recognised, both anchored on a ``*_ROOT`` name so an arbitrary
    expression elsewhere in the module is never mistaken for a library path:

    * ``os.path.join(_ACCEL_ROOT, "go", "neurons", "theta", "libtheta.so")``
    * ``_PACKAGE_ROOT / "accel" / "mojo" / "world_model" / "liblgssm.so"``
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
    ):
        root = next(
            (a.id for a in node.args if isinstance(a, ast.Name) and a.id.endswith("ROOT")),
            None,
        )
        if root is None:
            return None
        parts = [
            a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        ]
        return (root, parts) if parts else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        segments: list[str] = []
        cursor: ast.AST = node
        while isinstance(cursor, ast.BinOp) and isinstance(cursor.op, ast.Div):
            if not (isinstance(cursor.right, ast.Constant) and isinstance(cursor.right.value, str)):
                return None
            segments.append(cursor.right.value)
            cursor = cursor.left
        if isinstance(cursor, ast.Name) and cursor.id.endswith("ROOT"):
            return (cursor.id, list(reversed(segments)))
    return None


def _loader_output_paths(language: str, py_files: Sequence[Path], accel_root: Path) -> set[Path]:
    """Extract the absolute ``lib*.so`` paths the ctypes loaders open.

    These are the authoritative build destinations: the loader for a backend
    opens exactly the path this returns, so producing the library there is what
    makes the compiled-backend tests pass.
    """
    package_root = accel_root.parent
    outputs: set[Path] = set()
    for py_file in py_files:
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            matched = _loader_lib_parts(node)
            if matched is None:
                continue
            root, parts = matched
            if not (parts[-1].endswith(".so") and language in parts):
                continue
            base = accel_root if "ACCEL" in root else package_root
            outputs.add(base.joinpath(*parts))
    return outputs


def _iter_files(root: Path, suffix: str) -> list[Path]:
    """All ``*<suffix>`` files under ``root``, skipping vendored/build directories.

    Loaders live in several packages (``neurons/models``, ``accel``,
    ``world_model`` ...) and Mojo build recipes live in the ``.mojo`` sources,
    so the whole tree is scanned; pruning ``.pixi``/``__pycache__`` keeps it fast.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _PRUNED_DIRS]
        found.extend(Path(dirpath) / name for name in filenames if name.endswith(suffix))
    return sorted(found)


def discover_targets(
    language: str,
    *,
    accel_root: Path = _ACCEL_ROOT,
) -> list[BackendTarget]:
    """Discover every buildable backend for ``language`` ("go" or "mojo").

    The loader library-path expressions give the authoritative output directory
    and name (unambiguous), and the embedded build recipes give the matching
    source file name (``libhr.so`` <- ``hindmarsh_rose.go``). The source sits
    beside its library, so the two together pin one concrete build per backend.
    The whole package tree is scanned so loaders outside ``neurons/models`` /
    ``accel`` (e.g. ``world_model`` LGSSM) are discovered too.
    """
    py_files = _iter_files(accel_root.parent, ".py")
    pattern = _GO_HINT if language == "go" else _MOJO_HINT
    suffix = ".go" if language == "go" else ".mojo"
    # Recipes live both in the Python model modules and in the source-file header
    # comments (Mojo recipes such as ``libhr.so <- hindmarsh_rose.mojo`` only
    # appear beside the ``.mojo`` source), so read both for the source pairing.
    recipe_text = _read_text(py_files + _iter_files(accel_root, suffix))
    source_for_output = {out: src for src, out in _hint_pairs(recipe_text, pattern).items()}

    targets: list[BackendTarget] = []
    for output in sorted(_loader_output_paths(language, py_files, accel_root)):
        # Prefer the recipe pairing (needed for renamed outputs such as
        # libhr.so <- hindmarsh_rose.go); fall back to the lib<stem>.so naming
        # convention used by the primary neurons that carry no explicit recipe.
        candidates = []
        recipe_name = source_for_output.get(output.name)
        if recipe_name is not None:
            candidates.append(recipe_name)
        candidates.append(_conventional_source_name(output.name, suffix))
        source = next(
            (output.parent / c for c in candidates if (output.parent / c).is_file()), None
        )
        if source is None:
            continue
        targets.append(
            BackendTarget(
                language=language,
                name=source.name[: -len(suffix)],
                source=source,
                output=output,
            )
        )
    return targets


def _conventional_source_name(output_name: str, suffix: str) -> str:
    """``libcoba_lif.so`` -> ``coba_lif.go`` (strip the ``lib`` prefix and ``.so``)."""
    stem = output_name[: -len(".so")]
    if stem.startswith("lib"):
        stem = stem[len("lib") :]
    return stem + suffix


def _go_command(target: BackendTarget) -> list[str]:
    # Package mode ("." rather than the single source file) is the reproducible
    # build the loaders document: it compiles every non-test file in the
    # backend directory and emits the exact committed C header, so a
    # multi-file backend links and the generated header stays byte-stable.
    return [
        "go",
        "build",
        "-buildmode=c-shared",
        "-o",
        target.output.name,
        ".",
    ]


def _mojo_command(target: BackendTarget, mojo_command: Sequence[str]) -> list[str]:
    return [
        *mojo_command,
        "build",
        "--emit",
        "shared-lib",
        # Pin a portable ISA baseline (x86-64-v3 == AVX2/FMA/BMI, no AVX-512). Mojo's
        # --target-cpu defaults to the host CPU, so a build host that reports AVX-512 emits
        # AVX-512 mask ops (e.g. `kmovd`) into the kernel. A CI runner whose CPU lacks AVX-512F
        # then raises SIGILL (Illegal instruction) executing its own freshly-built library,
        # crashing the parity tests. x86-64-v3 is the actual capability ceiling of the hosted
        # GitHub runners (AVX2 present, AVX-512 absent), so it keeps 256-bit SIMD while running
        # everywhere. Production builds may override with a wider --target-cpu.
        "--target-cpu",
        "x86-64-v3",
        "-o",
        target.output.name,
        target.source.name,
    ]


def build_target(
    target: BackendTarget,
    *,
    mojo_command: Sequence[str] = ("mojo",),
    runner: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
) -> BuildResult:
    """Build one target in its source directory; return the outcome."""
    if target.language == "go":
        cmd = _go_command(target)
        env_note = "CGO_ENABLED=1"
    else:
        cmd = _mojo_command(target, mojo_command)
        env_note = ""
    run = runner if runner is not None else _default_runner
    try:
        completed = run(cmd, target.source.parent)
    except FileNotFoundError as exc:
        return BuildResult(target, False, f"toolchain missing: {exc}")
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-3:]
        return BuildResult(target, False, f"exit {completed.returncode}: {' | '.join(tail)}")
    if not target.output.is_file():
        return BuildResult(target, False, "build reported success but no library produced")
    return BuildResult(target, True, env_note or "ok")


def _default_runner(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if cmd[:1] == ["go"]:
        env["CGO_ENABLED"] = "1"
        # The accel go.mod pins a newer Go than CI's setup-go installs; force
        # toolchain auto-management so Go fetches the version go.mod requires
        # instead of failing under a GOTOOLCHAIN=local runner.
        env["GOTOOLCHAIN"] = "auto"
    return subprocess.run(  # noqa: S603 - fixed argv from trusted recipes, no shell
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=("go", "mojo", "all"),
        default="all",
        help="which backend language(s) to build",
    )
    parser.add_argument(
        "--mojo-command",
        default="mojo",
        help="mojo invocation (may be a pixi-run wrapper); shell-split",
    )
    parser.add_argument(
        "--require",
        default="",
        help="comma-separated backend names that MUST build (non-zero exit if not)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the requested backends and report; exit non-zero on required failures."""
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    languages = ("go", "mojo") if args.language == "all" else (args.language,)
    mojo_command = tuple(shlex.split(args.mojo_command))
    required = {name.strip() for name in args.require.split(",") if name.strip()}

    results: list[BuildResult] = []
    for language in languages:
        targets = discover_targets(language)
        print(f"[{language}] discovered {len(targets)} backend target(s)")
        for target in targets:
            result = build_target(target, mojo_command=mojo_command)
            results.append(result)
            status = "OK  " if result.ok else "FAIL"
            print(f"  {status} {language}:{target.name} -> {target.output.name} ({result.detail})")

    built = sum(1 for r in results if r.ok)
    failed = [r for r in results if not r.ok]
    print(f"built {built}/{len(results)} backend libraries; {len(failed)} failed")

    required_failures = [r for r in failed if r.target.name in required]
    if required_failures:
        names = ", ".join(sorted(r.target.name for r in required_failures))
        print(f"REQUIRED backends failed to build: {names}", file=sys.stderr)
        return 1
    if required:
        missing = required - {r.target.name for r in results}
        if missing:
            print(f"REQUIRED backends were never discovered: {sorted(missing)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
