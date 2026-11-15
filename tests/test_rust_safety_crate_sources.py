# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust safety crate source declaration tests

"""Every Rust file in the safety crate's folder is compiled, or named as unsupported source.

A ``.rs`` file that no compiled file declares is never built, and the tests inside it never
run, although the file looks like part of the crate. The crate's one module list is
``safety/lib.rs`` (``[lib] path`` in ``accel/rust/Cargo.toml``); the native crates in the
folder compile a few shared files through ``#[path]``. The only files left out are the seven
codec transcripts that the mirror-authority guide names as unsupported source, and
``safety/mod.rs``, which is not compiled either and therefore may declare no module: a
second module list there drifted from ``lib.rs`` while no build read it. It stays as a file
because the Rust documentation ratchet refuses a shrinking Rust source cohort.

The sources are read with their ``//`` and ``/* */`` comments removed, so a commented-out
declaration or ``#[path]`` counts for nothing (nested block comments are not modelled), and a
module declaration of ``lib.rs`` may not be conditional on a ``#[cfg]``, which could leave it
uncompiled. The guard covers the folder's top level: a sub-module file in a subfolder of a
declared module is outside its scope.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAFETY = ROOT / "src/sc_neurocore/accel/rust/safety"
GUIDE = ROOT / "docs/guides/accel_mirror_authority.md"

#: The codec transcripts kept as unsupported source, not compiled into the crate.
UNSUPPORTED_CODEC_TRANSCRIPTS = frozenset(
    {
        "aer_codec",
        "codec",
        "delta_codec",
        "entropy",
        "predictive_codec",
        "streaming_codec",
        "waveform_codec",
    }
)

#: Not compiled, kept for the Rust source cohort; it must hold no module list.
UNCOMPILED_POINTER = "mod.rs"

_MODULE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([a-z0-9_]+)\s*;", re.MULTILINE)
_INCLUDED = re.compile(r'#\[path\s*=\s*"([^"]+)"\]|include!\("([^"]+)"\)')
_CONDITIONAL = re.compile(
    r"#\[cfg\b[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+([a-z0-9_]+)\s*;"
)
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def _code(source: Path) -> str:
    """Return a Rust source's text without its ``//`` and ``/* */`` comments.

    Parameters
    ----------
    source : Path
        A Rust source file.

    Returns
    -------
    str
        The text with every comment replaced by a space.
    """
    return _COMMENT.sub(" ", source.read_text(encoding="utf-8"))


def _referenced(source: Path) -> set[Path]:
    """Return the files that one Rust source pulls in through ``#[path]`` or ``include!``.

    Parameters
    ----------
    source : Path
        A Rust source file.

    Returns
    -------
    set of Path
        The resolved files it names, relative to its own folder.
    """
    text = _code(source)
    return {
        (source.parent / (path or included)).resolve() for path, included in _INCLUDED.findall(text)
    }


def _compiled_files() -> set[Path]:
    """Return every file in the safety folder that a compiled crate reaches.

    The safety crate starts at ``lib.rs`` and reaches each module it declares; each native
    crate in the folder starts at its ``src`` files. Both follow ``#[path]`` and ``include!``.

    Returns
    -------
    set of Path
        The resolved files that some crate compiles.
    """
    lib = SAFETY / "lib.rs"
    declared = _MODULE.findall(_code(lib))
    frontier = [lib, *(SAFETY / f"{name}.rs" for name in declared)]
    frontier += sorted(SAFETY.glob("*_native/src/*.rs"))
    reached: set[Path] = set()
    while frontier:
        source = frontier.pop().resolve()
        if source in reached or not source.is_file():
            continue
        reached.add(source)
        frontier.extend(_referenced(source))
    return reached


def test_the_crate_declares_every_file_in_its_folder() -> None:
    """No ``.rs`` file beside ``lib.rs`` is left out, apart from the named transcripts."""
    compiled = _compiled_files()
    left_out = sorted(
        path.name
        for path in SAFETY.glob("*.rs")
        if path.resolve() not in compiled
        and path.stem not in UNSUPPORTED_CODEC_TRANSCRIPTS
        and path.name != UNCOMPILED_POINTER
    )
    assert left_out == []


def test_the_uncompiled_mod_rs_declares_no_second_module_list() -> None:
    """``safety/mod.rs`` is not compiled, so it may not hold a module list that drifts."""
    pointer = SAFETY / UNCOMPILED_POINTER
    assert pointer.resolve() not in _compiled_files()
    assert _MODULE.findall(_code(pointer)) == []


def test_every_declared_module_has_its_file() -> None:
    """Each module that ``lib.rs`` declares exists as a file in the folder."""
    declared = _MODULE.findall(_code(SAFETY / "lib.rs"))
    assert declared
    assert sorted(name for name in declared if not (SAFETY / f"{name}.rs").is_file()) == []


def test_the_unsupported_transcripts_are_the_ones_the_guide_names() -> None:
    """The transcripts exist, are not declared, and the guide names each of them."""
    declared = set(_MODULE.findall(_code(SAFETY / "lib.rs")))
    guide = GUIDE.read_text(encoding="utf-8")
    for name in sorted(UNSUPPORTED_CODEC_TRANSCRIPTS):
        assert (SAFETY / f"{name}.rs").is_file(), name
        assert name not in declared, name
        assert f"`{name}`" in guide, name


def test_no_module_declaration_of_lib_rs_is_conditional() -> None:
    """A ``#[cfg]`` before a declaration could leave its file uncompiled, so none is allowed."""
    assert _CONDITIONAL.findall(_code(SAFETY / "lib.rs")) == []
