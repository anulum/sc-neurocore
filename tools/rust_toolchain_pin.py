# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the Rust toolchain a measured crate pins

"""Read the Rust toolchain a crate pins and make every compiler call use it.

rustup chooses a toolchain from the ``RUSTUP_TOOLCHAIN`` variable, then from a
directory override, then from a ``rust-toolchain.toml`` or legacy
``rust-toolchain`` file in the working directory or one of its parents, then
from its default. A measurement that starts Cargo from the repository root
does not see a member crate's own toolchain file, and one that compiles a
revision in a temporary directory sees no directory override either; both
then run the host's default compiler.

This module reads the pin the way rustup would for the crate's own directory,
from the manifest's directory up to the repository root and never beyond it,
and returns the environment that makes every rustup proxy run that toolchain.
The environment also forbids rustup from installing a missing toolchain, so a
pin that does not run is refused instead of fetched.

Cargo does not compile through the rustup proxy when it is told another
compiler: ``RUSTC``, a wrapper in ``RUSTC_WRAPPER`` or
``RUSTC_WORKSPACE_WRAPPER``, their ``CARGO_BUILD_*`` forms, or ``build.rustc``
and its wrappers in a Cargo configuration file. With a pin, the environment
drops the first group, sets both wrappers to the empty value (which also turns
off a wrapper from a configuration file), and :func:`compiler_environment`
names the pinned toolchain's own compiler in ``RUSTC``, which takes precedence
over ``build.rustc``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import tomllib

__all__ = [
    "COMPILER_OVERRIDES",
    "LEGACY_FILE",
    "TOML_FILE",
    "RustToolchainPinError",
    "ToolchainPin",
    "compiler_environment",
    "pinned_toolchain",
    "reported_version_matches",
    "toolchain_environment",
]

#: The toolchain file rustup reads in its current form.
TOML_FILE = "rust-toolchain.toml"
#: The older file name, which holds a bare toolchain name or the same TOML.
LEGACY_FILE = "rust-toolchain"
#: A toolchain name as rustup spells channels, versions, dates and target triples.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
#: A numbered channel, optionally with a target triple: the release it names.
_NUMBERED = re.compile(r"(?P<release>\d+\.\d+(?:\.\d+)?)(?:-.+)?")
#: Variables through which a caller can make Cargo run another compiler; a pin removes them.
COMPILER_OVERRIDES = (
    "RUSTC",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
)
#: Wrapper variables a pin sets empty, which also turns off a configured wrapper.
_WRAPPERS = ("RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")


class RustToolchainPinError(ValueError):
    """Refuse a toolchain file that names no single usable toolchain."""


@dataclass(frozen=True)
class ToolchainPin:
    """The toolchain a crate pins and the repository-relative file that pins it.

    Attributes
    ----------
    channel : str
        The toolchain name as written, for example ``1.98.1`` or ``nightly-2026-08-18``.
    source : str
        Repository-relative POSIX path of the toolchain file.
    """

    channel: str
    source: str

    def to_public_dict(self) -> dict[str, str]:
        """Return the channel and the file that pins it, for a measurement record."""
        return {"channel": self.channel, "source": self.source}


def _text(path: Path, source: str) -> str:
    """Read one toolchain file as UTF-8 text, refusing other bytes."""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RustToolchainPinError(f"{source} is not UTF-8 text.") from exc


def _channel_from_toml(text: str, source: str) -> str:
    """Return the channel of a ``[toolchain]`` table, refusing a path or a missing name."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RustToolchainPinError(f"{source} is not valid TOML.") from exc
    table = document.get("toolchain")
    if isinstance(table, dict) and "path" in table:
        raise RustToolchainPinError(
            f"{source} names a toolchain by path; the measurement needs a named channel."
        )
    channel = table.get("channel") if isinstance(table, dict) else None
    if not isinstance(channel, str):
        raise RustToolchainPinError(f"{source} names no toolchain channel.")
    return channel


def _read_pin(path: Path, source: str) -> ToolchainPin:
    """Read and check the toolchain name one file pins."""
    text = _text(path, source)
    if path.name == TOML_FILE or text.lstrip().startswith("["):
        channel = _channel_from_toml(text, source)
    else:
        names = [line.strip() for line in text.splitlines() if line.strip()]
        if len(names) != 1:
            raise RustToolchainPinError(f"{source} must hold one toolchain name.")
        channel = names[0]
    if not _NAME.fullmatch(channel):
        raise RustToolchainPinError(
            f"{source} names the toolchain {channel!r}, which is not a toolchain name."
        )
    return ToolchainPin(channel, source)


def pinned_toolchain(root: Path, manifest: str) -> ToolchainPin | None:
    """Return the toolchain pinned nearest to a manifest within the repository.

    Parameters
    ----------
    root : Path
        Repository root; the search never leaves it.
    manifest : str
        Repository-relative path of the measured ``Cargo.toml``. It must already be
        known to lie inside the repository.

    Returns
    -------
    ToolchainPin or None
        The pin of the nearest directory, from the manifest's own up to the root,
        that holds a toolchain file; ``None`` when none does.

    Raises
    ------
    RustToolchainPinError
        If the nearest directory holds both files, or its file names no single
        named toolchain.
    """
    root = root.resolve()
    directory = PurePosixPath(manifest).parent
    while True:
        candidates = [
            name for name in (LEGACY_FILE, TOML_FILE) if (root / directory / name).is_file()
        ]
        if len(candidates) == 2:
            where = "the repository root" if directory == PurePosixPath(".") else str(directory)
            raise RustToolchainPinError(
                f"Two Rust toolchain files in {where} pin the measurement; keep one."
            )
        if candidates:
            source = (directory / candidates[0]).as_posix()
            return _read_pin(root / source, source)
        if directory == PurePosixPath("."):
            return None
        directory = directory.parent


def toolchain_environment(
    pin: ToolchainPin | None, environment: Mapping[str, str]
) -> dict[str, str]:
    """Return a copy of a process environment that runs the pinned toolchain.

    Parameters
    ----------
    pin : ToolchainPin or None
        The crate's pin, or ``None`` to run the caller's toolchain.
    environment : mapping of str to str
        The caller's environment; it is not changed.

    Returns
    -------
    dict of str to str
        With a pin: ``RUSTUP_TOOLCHAIN`` names it, ``RUSTUP_AUTO_INSTALL`` is ``0``,
        the variables of :data:`COMPILER_OVERRIDES` are removed and both wrapper
        variables are empty, whatever the caller set. Without a pin: an unchanged
        copy. Cargo still needs :func:`compiler_environment` to name the pinned
        compiler over a ``build.rustc`` setting.
    """
    result = dict(environment)
    if pin is not None:
        result["RUSTUP_TOOLCHAIN"] = pin.channel
        result["RUSTUP_AUTO_INSTALL"] = "0"
        for name in COMPILER_OVERRIDES:
            result.pop(name, None)
        for name in _WRAPPERS:
            result[name] = ""
    return result


def compiler_environment(environment: Mapping[str, str], rustc: Path) -> dict[str, str]:
    """Return a copy of an environment that makes Cargo run one compiler executable.

    Parameters
    ----------
    environment : mapping of str to str
        The environment to extend; it is not changed.
    rustc : Path
        The compiler executable, for a pin the toolchain's own ``bin/rustc``.

    Returns
    -------
    dict of str to str
        The copy with ``RUSTC`` naming the executable, which Cargo prefers to a
        ``build.rustc`` setting in a configuration file.
    """
    result = dict(environment)
    result["RUSTC"] = str(rustc)
    return result


def reported_version_matches(pin: ToolchainPin, reported: str) -> bool:
    """Tell whether ``rustc --version`` output belongs to a numbered pin.

    Parameters
    ----------
    pin : ToolchainPin
        The pinned toolchain.
    reported : str
        What ``rustc --version`` printed, for example ``rustc 1.98.1 (48a229cea 2026-09-01)``.

    Returns
    -------
    bool
        For a numbered channel, whether the compiler's version is exactly that
        release, or a release of that minor version when the channel names only two
        parts; a pre-release such as ``1.99.0-beta.3`` matches neither. A named
        channel such as ``stable`` or ``nightly-2026-08-18`` is resolved by rustup
        itself and always matches.
    """
    numbered = _NUMBERED.fullmatch(pin.channel)
    if numbered is None:
        return True
    words = reported.split()
    if len(words) < 2:
        return False
    release, version = numbered.group("release"), words[1]
    if release.count(".") == 1:
        return re.fullmatch(re.escape(release) + r"\.\d+", version) is not None
    return version == release
