# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Relative-interpreter wheel installation destination

"""Write relative POSIX launchers with normal wheel-installer RECORD entries."""

from __future__ import annotations

import io
import os
import re
import tokenize
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

from installer.destinations import SchemeDictionaryDestination
from installer.records import RecordEntry
from installer.scripts import Script
from installer.utils import Scheme

from tools.studio_desktop_wheel_catalog import SharedWheelPayload


@dataclass
class RelocatableWheelDestination(SchemeDictionaryDestination):
    """Install wheel scripts against the adjacent bundled Python executable.

    Linux delivery requires ``/usr/bin/env -S`` and ``/bin/sh``. The inherited installer writes all
    payload files and computes installed RECORD hashes from the actual bytes.
    The runtime must contain the selected Python next to the scripts; moving
    the whole runtime or invoking a symlink to a script preserves that binding.

    Attributes
    ----------
    shared_payloads : mapping of pathlib.Path to SharedWheelPayload
        Original identical library ownership derived from the complete selected
        closure. Sharing is confined to files created by this destination instance;
        it never grants permission to overwrite an existing file.
    """

    shared_payloads: Mapping[Path, SharedWheelPayload] = field(default_factory=dict)
    _created_shared_payloads: dict[Path, tuple[RecordEntry, tuple[int, int]]] = field(
        default_factory=dict, init=False, repr=False
    )

    def write_to_fs(
        self, scheme: Scheme, path: str, stream: BinaryIO, is_executable: bool
    ) -> RecordEntry:
        """Create a normal payload or retain an admitted file shared by its wheel owners.

        Parameters
        ----------
        scheme, path : str
            Standard installation scheme and original relative member path.
        stream : binary stream
            Original payload, or the declared transformed script bytes.
        is_executable : bool
            Original executable declaration.

        Returns
        -------
        installer.records.RecordEntry
            Actual bytes for this owner's normal installed RECORD. An identical
            shared file remains on the inode created by this destination.

        Raises
        ------
        ValueError
            Overwriting was requested, or shared bytes, mode or inode changed.
        FileExistsError
            An existing file was not created under this exact shared ownership plan.
        OSError
            A payload cannot be written or checked.
        """
        if self.overwrite_existing:
            raise ValueError("Desktop assembly never permits overwriting existing files.")
        target = self._path_with_destdir(scheme, path)
        if target.is_symlink():
            raise ValueError("A desktop payload destination is a symbolic link.")
        target = target.resolve()
        shared = self.shared_payloads.get(target)
        if shared is not None:
            if shared.executable != is_executable:
                raise ValueError("A shared wheel payload's executable declaration differs.")
            expected = RecordEntry.from_elements(path, shared.record_hash, str(shared.size_bytes))
            created = self._created_shared_payloads.get(target)
            if created is not None:
                record, inode = created
                current = target.stat()
                if (
                    (current.st_dev, current.st_ino) != inode
                    or current.st_nlink != 1
                    or bool(current.st_mode & 0o111) != shared.executable
                    or str(record.hash_) != shared.record_hash
                    or record.size != shared.size_bytes
                    or not expected.validate_stream(stream)
                ):
                    raise ValueError("A shared wheel payload differs from its admitted original.")
                with target.open("rb") as existing:
                    if not expected.validate_stream(existing):
                        raise ValueError("An installed shared wheel payload changed.")
                return RecordEntry(path, record.hash_, record.size)
        written = super().write_to_fs(scheme, path, stream, is_executable)
        if shared is not None:
            if str(written.hash_) != shared.record_hash or written.size != shared.size_bytes:
                raise ValueError("A new shared wheel payload differs from its original.")
            current = target.stat()
            self._created_shared_payloads[target] = (written, (current.st_dev, current.st_ino))
        return written

    def _launcher(self) -> bytes:
        if self.script_kind != "posix":
            raise ValueError("Desktop runtime launchers require a POSIX destination.")
        executable = Path(self.interpreter).name
        if re.fullmatch(r"python(?:3(?:\.[0-9]+t?)?)?", executable) is None:
            raise ValueError("The bundled interpreter must have a Python executable name.")
        # A comment-only shebang preserves original docstrings and future imports.
        # https://www.gnu.org/s/coreutils/manual/html_node/env-invocation.html
        return (
            "#!/usr/bin/env -S /bin/sh -c 'exec "
            '"$(dirname -- "$(realpath -- "$0")")"/'
            f'{executable} "$0" "$@"\'\n'
        ).encode("ascii")

    def write_script(
        self, name: str, module: str, attr: str, section: Literal["console", "gui"]
    ) -> RecordEntry:
        """Install a declared console or GUI entry point with a relative interpreter.

        Parameters
        ----------
        name, module, attr : str
            Original wheel entry-point name, module and callable attribute.
        section : str
            Wheel entry-point kind, either console or GUI.

        Returns
        -------
        installer.records.RecordEntry
            Actual generated launcher path, hash and length for final RECORD.

        Raises
        ------
        ValueError
            The destination platform or bundled interpreter name is unsupported.
        OSError
            The installer cannot write the target script.
        """
        script_name, generated = Script(name, module, attr, section).generate("/python", "posix")
        _, _, body = generated.partition(b"\n")
        with io.BytesIO(self._launcher() + body) as stream:
            return self.write_to_fs(Scheme("scripts"), script_name, stream, True)

    def write_file(
        self,
        scheme: Scheme,
        path: str | os.PathLike[str],
        stream: BinaryIO,
        is_executable: bool,
    ) -> RecordEntry:
        """Preserve payload encodings when replacing generic Python shebangs.

        Parameters
        ----------
        scheme : str
            Standard wheel installation scheme.
        path : str or pathlib.Path
            Wheel destination relative to that scheme.
        stream : binary stream
            Original wheel member bytes.
        is_executable : bool
            Original executable declaration, retained for ordinary payloads.

        Returns
        -------
        installer.records.RecordEntry
            Actual installed bytes for the standard installer's RECORD writer.

        Raises
        ------
        ValueError
            A Python script requests an unsupported destination or interpreter.
        SyntaxError
            A generic Python script declares an invalid source encoding.
        OSError
            Reading or writing the actual wheel member fails.
        """
        if scheme != "scripts":
            return self.write_to_fs(scheme, os.fspath(path), stream, is_executable)
        data = stream.read()
        first, _, body = data.partition(b"\n")
        if first.rstrip(b"\r") in {b"#!python", b"#!pythonw"}:
            tokenize.detect_encoding(io.BytesIO(data).readline)
            data = self._launcher() + body
        with io.BytesIO(data) as content:
            return self.write_to_fs(scheme, os.fspath(path), content, is_executable)
