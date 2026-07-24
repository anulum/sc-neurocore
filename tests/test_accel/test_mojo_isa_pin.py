# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — enforce the portable Mojo ISA baseline at every call site
"""Unit-test ``pin_isa`` and statically forbid unpinned ``mojo build``/``run``.

The static guard is the drift backstop: any newly added ``mojo build`` or
``mojo run`` subprocess argv that is not routed through ``pin_isa`` (and so does
not carry ``--target-cpu x86-64-v3``) fails here, before it can SIGILL a
non-AVX-512 CI leg.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from sc_neurocore.accel.mojo.isa_baseline import MOJO_TARGET_CPU, pin_isa

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Adjacent-token argv shapes that name an actual Mojo build/run subprocess. The
# optional ``)`` also catches a resolver-call executable such as
# ``_require_tool("mojo"), "build"`` / ``shutil.which("mojo"), "run"`` — an
# absolute-path shape that the bare-token pattern would miss (and that must still
# be routed through pin_isa so it carries --target-cpu).
_MOJO_ARGV = re.compile(r'"mojo"\)?,\s*"(?:build|run)"|"run",\s*"mojo",\s*"(?:build|run)"')

# Call sites where the token pair is a monkeypatched stub, not a real invocation:
# ``_default_runner`` never launches Mojo (subprocess.run is patched) and the real
# tools/build_accel_backends.py pins --target-cpu itself. The focused suite split
# of the build-accel tests keeps the same monkeypatched shape.
_ALLOWLIST = {
    "tests/test_tools/test_build_accel_backends.py",
    "tests/test_tools/test_build_accel_backends_commands_and_build.py",
}


def test_pin_isa_inserts_baseline_after_build() -> None:
    assert pin_isa(["mojo", "build", "-o", "x.so", "x.mojo"]) == [
        "mojo",
        "build",
        "--target-cpu",
        MOJO_TARGET_CPU,
        "-o",
        "x.so",
        "x.mojo",
    ]


def test_pin_isa_inserts_baseline_after_run_through_pixi() -> None:
    assert pin_isa(["pixi", "run", "mojo", "run", "k.mojo"]) == [
        "pixi",
        "run",
        "mojo",
        "run",
        "--target-cpu",
        MOJO_TARGET_CPU,
        "k.mojo",
    ]


def test_pin_isa_pins_absolute_path_mojo_executable() -> None:
    # shutil.which("mojo") / _require_tool("mojo") resolves to an absolute path;
    # it must be pinned exactly like the bare token or the built kernel SIGILLs.
    argv = ["/home/anulum/.local/bin/mojo", "build", "--emit", "shared-lib", "x.mojo"]
    assert pin_isa(argv) == [
        "/home/anulum/.local/bin/mojo",
        "build",
        "--target-cpu",
        MOJO_TARGET_CPU,
        "--emit",
        "shared-lib",
        "x.mojo",
    ]


def test_pin_isa_pins_resolved_mojo_run_through_pixi() -> None:
    argv = ["/opt/pixi/bin/pixi", "run", "/usr/bin/mojo", "run", "k.mojo"]
    assert pin_isa(argv) == [
        "/opt/pixi/bin/pixi",
        "run",
        "/usr/bin/mojo",
        "run",
        "--target-cpu",
        MOJO_TARGET_CPU,
        "k.mojo",
    ]


def test_pin_isa_ignores_mojo_source_file_basename() -> None:
    # A ``*.mojo`` source path has basename ``x.mojo`` (not ``mojo``) and is not
    # followed by a subcommand, so it must never be mistaken for the executable.
    argv = ["python", "compile.py", "kernels/x.mojo", "build"]
    assert pin_isa(argv) == argv


def test_pin_isa_is_idempotent() -> None:
    once = pin_isa(["mojo", "run", "k.mojo"])
    assert pin_isa(once) == once


def test_pin_isa_leaves_non_mojo_argv_untouched() -> None:
    argv = ["go", "build", "-o", "lib.so", "x.go"]
    assert pin_isa(argv) == argv


def test_pin_isa_does_not_mutate_input() -> None:
    argv = ["mojo", "build", "x.mojo"]
    pin_isa(argv)
    assert argv == ["mojo", "build", "x.mojo"]


def _tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [_REPO_ROOT / relative for relative in result.stdout.split("\0") if relative]


def _enclosing_list(text: str, pos: int) -> tuple[int, int]:
    """Return (open, close) indices of the ``[...]`` that encloses ``pos``."""
    depth = 0
    start = pos
    while start >= 0:
        char = text[start]
        if char == "]":
            depth += 1
        elif char == "[":
            if depth == 0:
                break
            depth -= 1
        start -= 1
    depth = 0
    quote = ""
    end = start
    while end < len(text):
        char = text[end]
        if quote:
            if char == quote and text[end - 1] != "\\":
                quote = ""
        elif char in ("'", '"'):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                break
        end += 1
    return start, end


def test_every_mojo_build_run_argv_is_pinned() -> None:
    """No unpinned ``mojo build``/``run`` argv may exist outside the allowlist."""
    self_rel = Path(__file__).resolve().relative_to(_REPO_ROOT).as_posix()
    offenders: list[str] = []
    for path in _tracked_python_files():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in _ALLOWLIST or rel == self_rel:
            continue
        text = path.read_text(encoding="utf-8")
        for match in _MOJO_ARGV.finditer(text):
            open_idx, close_idx = _enclosing_list(text, match.start())
            if open_idx < 0 or close_idx >= len(text):
                continue
            list_text = text[open_idx : close_idx + 1]
            if text[:open_idx].rstrip().endswith("pin_isa(") or "--target-cpu" in list_text:
                continue
            lineno = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, "unpinned mojo build/run argv (wrap with pin_isa):\n" + "\n".join(
        offenders
    )
