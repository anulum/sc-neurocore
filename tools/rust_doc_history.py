# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — original Rust source cohort and individual debt protection

"""Protect immutable Git Rust cohorts, original compiler cases and lowered allowances."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import cast

from tools.docstring_policy_git import baseline_commit, git_bytes
from tools.docstring_policy_scope import PolicyError
from tools.rust_doc_measurement import (
    EXPANSION_PREFIX,
    RustDebtCase,
    RustMeasurement,
    RustMeasurementError,
    measure_rust_findings,
)
from tools.rust_doc_symbols import NativeParser, build_parser

LEGACY_SCHEMA = "sc-neurocore.rust-doc-ceiling.v1"
INDIVIDUAL_SCHEMA = "sc-neurocore.rust-doc-ceiling.v2"
#: Name of the persistent folder of the original source image in the temporary directory.
ORIGINAL_IMAGE_DIR = "sc-neurocore-rust-doc-original"
#: Environment variable that places the original source image elsewhere.
ORIGINAL_IMAGE_ENV = "SC_NEUROCORE_RUST_DOC_ORIGINAL_DIR"
#: File that marks a folder as the ratchet's own; only a marked folder is ever emptied.
IMAGE_MARKER = ".sc-neurocore-rust-doc-original"
#: Entries of the image that survive between runs: Cargo's build outputs, never source.
_BUILD_OUTPUTS = frozenset({"target", "native-target"})
#: Whether runs take an exclusive lock and check the folder's owner (POSIX).
_LOCKED = os.name == "posix"


def original_image_directory(root: Path) -> Path:
    """Return the persistent folder that holds the original source image.

    The image is extracted afresh on every run, but its folder and Cargo's
    build outputs in it persist. Cargo therefore reuses the dependencies it
    built before, Z3 from C++ source among them, instead of building them
    cold in a new temporary directory on every run. A cold build could take
    longer than the measurement's own timeout on a loaded host. The folder
    stays outside the checkout, so no rustup directory override or toolchain
    file of the checkout reaches the original revision.

    Parameters
    ----------
    root : Path
        The repository being measured; the folder may not lie inside it.

    Returns
    -------
    Path
        The folder named by ``SC_NEUROCORE_RUST_DOC_ORIGINAL_DIR``, else
        ``sc-neurocore-rust-doc-original`` in the temporary directory.

    Raises
    ------
    RustMeasurementError
        If the override names a folder inside the repository.
    """
    named = os.environ.get(ORIGINAL_IMAGE_ENV)
    if not named:
        return Path(tempfile.gettempdir()) / ORIGINAL_IMAGE_DIR
    folder = Path(named).resolve()
    if folder.is_relative_to(root.resolve()):
        raise RustMeasurementError(
            f"{ORIGINAL_IMAGE_ENV} names {folder}, inside the measured checkout; "
            "the original revision is measured in a folder outside it."
        )
    return folder


def _private_folder(image: Path) -> None:
    """Create the image folder for this account alone, and refuse one another account could alter."""
    image.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.mkdir(mode=0o700)
    except FileExistsError:
        pass
    status = os.lstat(image)
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise RustMeasurementError(f"The original source image folder {image} is not a directory.")
    if _LOCKED and (status.st_uid != os.getuid() or status.st_mode & (stat.S_IWGRP | stat.S_IWOTH)):
        raise RustMeasurementError(
            f"The original source image folder {image} must belong to this account and be "
            "writable by it alone; it is not used."
        )


@contextmanager
def _exclusive(image: Path) -> Iterator[None]:
    """Hold the image for one measurement; a concurrent run says so and waits for its turn."""
    _private_folder(image)
    if not _LOCKED:
        yield
        return
    import fcntl

    descriptor = os.open(image, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"Waiting for another Rust documentation run, which holds {image}.",
                file=sys.stderr,
                flush=True,
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _reset_image(image: Path) -> None:
    """Empty the ratchet's own folder except its build outputs, and mark it.

    Everything but real ``target`` and ``native-target`` folders and the
    marker is deleted. A folder that holds anything and lacks the marker is
    not the ratchet's and is refused, so an override cannot empty a folder
    of someone else's.
    """
    marker = image / IMAGE_MARKER
    entries = list(image.iterdir())
    if entries and (marker.is_symlink() or not marker.is_file()):
        raise RustMeasurementError(
            f"{image} holds files but no {IMAGE_MARKER} marker, so it is not the ratchet's "
            "folder; it is not emptied. Name an empty folder of its own."
        )
    for entry in entries:
        if entry.name == IMAGE_MARKER:
            continue
        if entry.name in _BUILD_OUTPUTS and entry.is_dir() and not entry.is_symlink():
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    marker.write_text("The Rust documentation ratchet's original source image.\n")


@dataclass(frozen=True)
class ProtectedRustDebt:
    """Bind independently measured original compiler debt to an immutable Git revision."""

    revision: str
    original: RustMeasurement
    original_ceiling: int | None
    original_ceiling_sha256: str | None

    def to_public_dict(self) -> dict[str, object]:
        """Serialize the actual original measurement, Git object and ceiling hash."""
        return {
            "git_revision": self.revision,
            "original": self.original.to_public_dict(),
            "original_ceiling": self.original_ceiling,
            "original_ceiling_sha256": self.original_ceiling_sha256,
        }


def _document(raw: bytes) -> dict[str, object]:
    """Validate the nonnegative scalar and admitted original ceiling protocols."""
    try:
        data: object = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RustMeasurementError("The Rust ceiling must contain valid JSON.") from exc
    if not isinstance(data, dict):
        raise RustMeasurementError("The Rust ceiling must be a JSON object.")
    result = cast(dict[str, object], data)
    count = result.get("undocumented")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise RustMeasurementError("The Rust ceiling must contain a nonnegative integer.")
    if result.get("schema_version") not in {None, LEGACY_SCHEMA, INDIVIDUAL_SCHEMA}:
        raise RustMeasurementError("The Rust ceiling schema is unsupported.")
    return result


def _cases(document: dict[str, object]) -> frozenset[RustDebtCase] | None:
    """Read unique individual allowances without authorizing invented declarations."""
    if document.get("schema_version") != INDIVIDUAL_SCHEMA:
        return None
    provenance = document.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("debt_cases"), list):
        raise RustMeasurementError("An individual Rust ceiling requires a case array.")
    cases: set[RustDebtCase] = set()
    for entry in provenance["debt_cases"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"source", "identity"}
            or any(not isinstance(value, str) or not value for value in entry.values())
        ):
            raise RustMeasurementError("The Rust ceiling case identity shape is invalid.")
        case = RustDebtCase(entry["source"], entry["identity"])
        if case in cases:
            raise RustMeasurementError("Rust ceiling declaration cases must be unique.")
        cases.add(case)
    if len(cases) != document["undocumented"]:
        raise RustMeasurementError("Rust ceiling cases disagree with their scalar.")
    return frozenset(cases)


def _source_image(root: Path, revision: str, names: list[str], image: Path) -> None:
    """Extract immutable original build inputs and index a separate native source image."""
    selected = [
        name
        for name in names
        if name.endswith((".rs", ".toml", ".lock", ".md", ".wgsl")) or "/.cargo/" in "/" + name
    ]
    archive = git_bytes(root, "archive", "--format=tar", revision, "--", *selected)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for name in selected:
            member = stream.getmember(name)
            if (
                not member.isfile()
                or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
            ):
                raise RustMeasurementError(
                    "Original Rust build sources must be regular local files."
                )
            source = stream.extractfile(member)
            if source is None:
                raise RustMeasurementError("An original Rust source could not be read.")
            path = image / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read())
    # Cargo writes build outputs, including generated Rust, below the image unless
    # the caller redirects them; neither default nor test location is source.
    (image / ".gitignore").write_text(f"native-target/\ntarget/\n{IMAGE_MARKER}\n")
    commands = [
        ["git", "init", "-q", "--initial-branch=main"],
        ["git", "add", "--", "."],
        [
            "git",
            "-c",
            "user.name=Native baseline",
            "-c",
            "user.email=native@example.invalid",
            "commit",
            "-qm",
            "immutable original Rust source image",
        ],
    ]
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        env.pop(key, None)
    for argv in commands:
        result = subprocess.run(
            argv, cwd=image, env=env, capture_output=True, check=False, timeout=30
        )
        if result.returncode != 0:
            raise RustMeasurementError("The original Rust source image could not be indexed.")


def protect_rust_debt(
    root: Path,
    current: RustMeasurement,
    ceiling: Path,
    *,
    allow_create: bool = False,
    parser: NativeParser | None = None,
) -> ProtectedRustDebt:
    """Refuse original cohort loss, changed manifest scope and new undocumented cases.

    Parameters
    ----------
    root : Path
        Exact repository whose original Git source determines debt allowances.
    current : RustMeasurement
        Actual current compiler measurement and native syntax identities.
    ceiling : Path
        Candidate ceiling, which cannot grant new original debt allowances.
    allow_create : bool
        Permit explicit initialization from independently measured original debt.
    parser : NativeParser or None
        Verified producer reused for original source measurement.

    Returns
    -------
    ProtectedRustDebt
        Immutable original source and independently reproduced compiler debt.

    Raises
    ------
    RustMeasurementError
        If original scope, native measurement, ceiling or declaration allowances fail.

    Notes
    -----
    Original macro-generated debt is measured under an expansion identity, because
    the baseline may predate the rule that refuses it. The current source gets no
    such identity: an undocumented macro-generated declaration is refused there,
    so that original debt can only be repaid by documenting the macro. The syntax
    parser cannot see expanded declarations, so their retention is not checked.

    Legacy aggregate ceilings are upper bounds: documentation-only commits
    can reduce findings without rewriting that bound. New debt is still checked
    against the independently measured original cases. Individual-case records
    must reproduce their recorded counts and declaration identities exactly.
    """
    try:
        revision = baseline_commit(root, os.environ)
        names = [
            name.decode("utf-8")
            for name in git_bytes(root, "ls-tree", "-r", "-z", "--name-only", revision).split(b"\0")
            if name
        ]
        original_paths = {name for name in names if name.endswith(".rs")}
        if not original_paths <= set(current.paths):
            raise RustMeasurementError("The original Rust source cohort must not shrink.")
        expected_manifest = "engine/Cargo.toml" if "engine/Cargo.toml" in names else "Cargo.toml"
        if current.manifest != expected_manifest:
            raise RustMeasurementError("The original Rust library manifest must not change.")
        if "engine/missing_docs_ceiling.json" in names:
            ceiling_name = "engine/missing_docs_ceiling.json"
        else:
            try:
                ceiling_name = ceiling.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                ceiling_name = ""
        original_raw = (
            git_bytes(root, "show", revision + ":" + ceiling_name)
            if ceiling_name in names
            else None
        )
        original_document = _document(original_raw) if original_raw is not None else None
        image = original_image_directory(root)
        with _exclusive(image):
            _reset_image(image)
            _source_image(root, revision, names, image)
            original = measure_rust_findings(
                image, current.manifest, parser=parser or build_parser(), admit_expansions=True
            )
    except (
        PolicyError,
        OSError,
        UnicodeError,
        tarfile.TarError,
        KeyError,
        subprocess.SubprocessError,
    ) as exc:
        raise RustMeasurementError(
            "The original Rust source revision could not be qualified."
        ) from exc
    original_ceiling = (
        cast(int, original_document["undocumented"]) if original_document is not None else None
    )
    if original_document is not None:
        if original.undocumented > cast(int, original_ceiling) or (
            original_document.get("schema_version") == INDIVIDUAL_SCHEMA
            and original.undocumented != original_ceiling
        ):
            raise RustMeasurementError(
                "The original Rust native measurement does not reproduce its ceiling: "
                f"measured={original.undocumented}, recorded={original_ceiling}, "
                f"revision={revision}, compiler={original.rustc_version}."
            )
        original_cases = _cases(original_document)
        if original_cases is not None and original_cases != original.cases:
            raise RustMeasurementError(
                "The original Rust native cases do not reproduce their ceiling."
            )
        provenance = original_document.get("provenance")
        if (
            isinstance(provenance, dict)
            and "rustc" in provenance
            and provenance["rustc"] != original.rustc_version
        ):
            raise RustMeasurementError(
                "The native Rust version differs from the original ceiling producer."
            )
    if allow_create and not ceiling.exists():
        candidate: dict[str, object] = {"undocumented": original.undocumented}
    else:
        try:
            candidate = _document(ceiling.read_bytes())
        except OSError as exc:
            raise RustMeasurementError("No Rust documentation ceiling is available.") from exc
    if original_ceiling is not None and cast(int, candidate["undocumented"]) > original_ceiling:
        raise RustMeasurementError(
            "The candidate Rust ceiling must not exceed its original Git ceiling."
        )
    candidate_cases = _cases(candidate)
    if (
        original_document is not None
        and _cases(original_document) is not None
        and candidate_cases is None
    ):
        raise RustMeasurementError(
            "An individual Rust ceiling cannot revert to an aggregate protocol."
        )
    if candidate_cases is not None and not candidate_cases <= original.cases:
        raise RustMeasurementError("The candidate Rust ceiling adds original debt allowances.")
    new_cases = current.cases - (original.cases if candidate_cases is None else candidate_cases)
    if new_cases:
        raise RustMeasurementError(
            "New undocumented Rust declarations are not allowed: "
            + "; ".join(case.source + ":" + case.identity for case in sorted(new_cases))
        )
    retained = {case for case in original.cases if not case.identity.startswith(EXPANSION_PREFIX)}
    if not retained <= current.declarations:
        raise RustMeasurementError(
            "Original undocumented Rust declarations must remain in the source cohort."
        )
    current.verify(root)
    return ProtectedRustDebt(
        revision,
        original,
        original_ceiling,
        hashlib.sha256(original_raw).hexdigest() if original_raw is not None else None,
    )
