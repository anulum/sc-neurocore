# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Versioned per-user desktop installation

"""Install immutable desktop payloads while retaining the user's research profile."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.bundle import BundleManifest, verify_bundle
from sc_neurocore.studio.desktop.configuration import read_configuration
from sc_neurocore.studio.desktop.installation_registration import DesktopRegistration
from sc_neurocore.studio.desktop.paths import DesktopPaths
from sc_neurocore.studio.workspace_schema import write_atomic

_OWNER = '{"schema_version":"studio.desktop.installation.v1"}\n'
_STAGE = '{"schema_version":"studio.desktop.install-stage.v1"}\n'
_DATA_SCHEMA = "studio.desktop.installed-data.v1"


class DesktopInstallation:
    """Manage the code installation without migrating or removing user data.

    Attributes
    ----------
    profile : DesktopPaths
        Existing per-user configuration, projects, state and cache locations.
    root : pathlib.Path
        Application-only directory containing complete versioned bundles.
    registration : DesktopRegistration
        Owned command launchers and freedesktop menu entry.
    """

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        """Resolve a profile and its managed code location without writing files.

        Parameters
        ----------
        environ : mapping of str to str, optional
            User home/XDG configuration, or the actual process environment.
        """
        self.profile = DesktopPaths.from_environment(environ)
        self.root = self.profile.data / "application"
        self.registration = DesktopRegistration(self.root, self.profile)

    def install(self, source: Path) -> BundleManifest:
        """Copy, verify and activate a self-contained bundle at its content identity.

        Parameters
        ----------
        source : pathlib.Path
            Complete extracted bundle, carrying its full runtime and manifest.

        Returns
        -------
        BundleManifest
            Installed and selected version. Identical installations are idempotent.

        Raises
        ------
        DesktopStartRefused
            The payload, ownership, retained data contract or profile lock refuses.
        OSError
            Copying, registration or atomic activation fails. The previous selected
            version remains selected until the final atomic symlink replacement.
        """
        source = source.resolve()
        manifest = verify_bundle(source)
        target = self.root / "versions" / manifest.build_id
        if self.root.resolve().is_relative_to(source) or (
            source.is_relative_to(self.root) and source != target
        ):
            raise DesktopStartRefused("The bundle and installation must be separate directories.")
        with self._locked():
            managed = self._owned()
            self.registration.verify(managed=managed)
            self._compatible(manifest)
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not managed:
                write_atomic(self.root / "owner.json", _OWNER)
            versions = self.root / "versions"
            versions.mkdir(exist_ok=True, mode=0o700)
            self._recover()
            self._versions()
            if target.exists():
                if verify_bundle(target) != manifest:
                    raise DesktopStartRefused("The installed bundle identity does not match.")
            else:
                stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=versions))
                try:
                    write_atomic(stage / "owner.json", _STAGE)
                    shutil.copytree(source, stage / "payload", symlinks=True)
                    if verify_bundle(stage / "payload") != manifest:
                        raise DesktopStartRefused("The copied desktop bundle does not match.")
                    os.replace(stage / "payload", target)
                finally:
                    shutil.rmtree(stage)
            self._retain_data_contract(manifest)
            self.registration.register(managed=True)
            self._activate(manifest.build_id)
        return manifest

    def rollback(self, build_id: str) -> BundleManifest:
        """Select a complete retained version only under the same data contract.

        Parameters
        ----------
        build_id : str
            Full content identity of a previously installed bundle.

        Returns
        -------
        BundleManifest
            Verified selected bundle; user configuration and experiments unchanged.

        Raises
        ------
        DesktopStartRefused
            The version is absent, modified, incompatible or the profile is busy.
        OSError
            Reading the version, registering launch files or activation fails.
        """
        if re.fullmatch(r"[0-9a-f]{64}", build_id) is None:
            raise DesktopStartRefused("Rollback requires a complete installed build identity.")
        with self._locked():
            if not self._owned():
                raise DesktopStartRefused("SNN Studio is not installed for this profile.")
            self._recover()
            if build_id not in self._versions():
                raise DesktopStartRefused("That Studio version is not installed.")
            manifest = verify_bundle(self.root / "versions" / build_id)
            if manifest.build_id != build_id:
                raise DesktopStartRefused("The retained Studio version identity does not match.")
            self._compatible(manifest)
            self.registration.register(managed=True)
            self._activate(build_id)
        return manifest

    def uninstall(self) -> None:
        """Remove verified application code and launch files, preserving the profile.

        Raises
        ------
        DesktopStartRefused
            Ownership, payload integrity or a running backend prevents removal.
        OSError
            Inspecting or removing the verified application files fails.

        Notes
        -----
        Projects, desktop configuration, browser sessions, job/identity stores,
        evidence, cache and retained data compatibility are never removed.
        """
        with self._locked():
            if not self._owned():
                raise DesktopStartRefused("SNN Studio is not installed for this profile.")
            self._recover()
            self._current()
            for build_id in self._versions():
                manifest = verify_bundle(self.root / "versions" / build_id)
                if manifest.build_id != build_id:
                    raise DesktopStartRefused("An installed version has an unexpected identity.")
            self.registration.verify(managed=True)
            self.registration.remove()
            shutil.rmtree(self.root)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.profile.prepare()
        with (
            (self.profile.state / "install.lock").open("a+b") as install,
            (self.profile.state / "backend.lock").open("a+b") as backend,
        ):
            try:
                fcntl.flock(install, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(backend, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DesktopStartRefused(
                    "Close Studio and finish any other installation before changing its application."
                ) from None
            yield

    def _owned(self) -> bool:
        if self.root.is_symlink():
            raise DesktopStartRefused("The Studio application directory cannot be a symlink.")
        if not self.root.exists() or (self.root.is_dir() and not any(self.root.iterdir())):
            return False
        if self._record(self.root / "owner.json") != _OWNER.encode():
            raise DesktopStartRefused(
                "The application directory is not a managed Studio installation."
            )
        if {path.name for path in self.root.iterdir()} - {
            "owner.json",
            "versions",
            "current",
            ".next",
        }:
            raise DesktopStartRefused("The application directory contains unowned files.")
        return True

    def _versions(self) -> tuple[str, ...]:
        path = self.root / "versions"
        if not path.exists():
            return ()
        if path.is_symlink() or not path.is_dir():
            raise DesktopStartRefused("The installed version directory is invalid.")
        versions = tuple(sorted(path.iterdir()))
        if any(
            not version.is_dir()
            or version.is_symlink()
            or re.fullmatch(r"[0-9a-f]{64}", version.name) is None
            for version in versions
        ):
            raise DesktopStartRefused("The installed version directory contains unowned entries.")
        return tuple(version.name for version in versions)

    def _current(self) -> BundleManifest | None:
        path = self.root / "current"
        if not path.is_symlink():
            if path.exists():
                raise DesktopStartRefused("The Studio application selector is not a managed link.")
            return None
        target = os.readlink(path)
        if re.fullmatch(r"versions/[0-9a-f]{64}", target) is None:
            raise DesktopStartRefused(
                "The Studio application selector leaves its version directory."
            )
        manifest = verify_bundle(self.root / target)
        if manifest.build_id != Path(target).name:
            raise DesktopStartRefused("The selected Studio version identity does not match.")
        return manifest

    def _compatible(self, manifest: BundleManifest) -> None:
        read_configuration(self.profile.config / "desktop.json")
        stamp = self.profile.state / "application-data.json"
        if stamp.is_symlink():
            raise DesktopStartRefused("The retained application data contract cannot be a symlink.")
        if stamp.exists():
            try:
                data: object = json.loads(self._record(stamp))
            except (ValueError, UnicodeError):
                raise DesktopStartRefused(
                    "The retained application data contract is invalid."
                ) from None
            if not isinstance(data, dict) or data != {
                "schema_version": _DATA_SCHEMA,
                "compatibility_key": manifest.compatibility_key,
            }:
                raise DesktopStartRefused(
                    "This package requires a separately qualified data migration."
                )
        current = self._current()
        if current is not None and current.compatibility_key != manifest.compatibility_key:
            raise DesktopStartRefused("This version does not share the installed data contract.")

    def _retain_data_contract(self, manifest: BundleManifest) -> None:
        path = self.profile.state / "application-data.json"
        if not path.exists():
            write_atomic(
                path,
                json.dumps(
                    {
                        "schema_version": _DATA_SCHEMA,
                        "compatibility_key": manifest.compatibility_key,
                    },
                    sort_keys=True,
                )
                + "\n",
            )

    def _recover(self) -> None:
        versions = self.root / "versions"
        if versions.is_symlink():
            raise DesktopStartRefused("The installed version directory cannot be a symlink.")
        for path in versions.glob(".stage-*"):
            if (
                path.is_symlink()
                or not path.is_dir()
                or self._record(path / "owner.json") != _STAGE.encode()
            ):
                raise DesktopStartRefused(
                    "An unfinished application directory has no Studio owner."
                )
            shutil.rmtree(path)
        pending = self.root / ".next"
        if pending.is_symlink():
            if re.fullmatch(r"versions/[0-9a-f]{64}", os.readlink(pending)) is None:
                raise DesktopStartRefused(
                    "The pending application selector is not owned by Studio."
                )
            pending.unlink()
        elif pending.exists():
            raise DesktopStartRefused("The pending application selector contains an unowned file.")

    def _activate(self, build_id: str) -> None:
        pending = self.root / ".next"
        try:
            pending.symlink_to("versions/" + build_id)
            os.replace(pending, self.root / "current")
        finally:
            pending.unlink(missing_ok=True)

    @staticmethod
    def _record(path: Path) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise DesktopStartRefused("An installation record must be a regular file.")
        with path.open("rb") as stream:
            value = stream.read(8193)
        if len(value) > 8192:
            raise DesktopStartRefused("An installation record exceeds its size limit.")
        return value
