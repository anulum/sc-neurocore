# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-user desktop launch registration

"""Register only owned user launchers and a freedesktop application entry."""

from __future__ import annotations

import os
import shlex
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.desktop.paths import DesktopPaths

_HEADER = """# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed SNN Studio launcher
"""


@dataclass(frozen=True, slots=True)
class DesktopRegistration:
    """The launch files belonging to one managed installation and user profile.

    Attributes
    ----------
    root : pathlib.Path
        Managed versioned application directory, separate from project storage.
    profile : DesktopPaths
        Validated home/XDG profile selecting user launch and menu locations.
    """

    root: Path
    profile: DesktopPaths

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return the application launcher, management launcher and desktop entry."""
        return tuple(self._files())

    def verify(self, *, managed: bool) -> None:
        """Refuse an existing foreign or modified launch file before changing any.

        Parameters
        ----------
        managed : bool
            Whether the caller verified the installation's ownership record.

        Raises
        ------
        DesktopStartRefused
            A launch path is a symlink, non-regular, unowned or contains modified bytes.
        OSError
            An existing launch file cannot be read.
        """
        for path, (expected, _mode) in self._files().items():
            if path.is_symlink():
                raise DesktopStartRefused(f"Studio will not replace the launch symlink: {path}")
            try:
                with open(
                    path,
                    "rb",
                    opener=lambda name, flags: os.open(name, flags | os.O_NONBLOCK | os.O_NOFOLLOW),
                ) as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise DesktopStartRefused(
                            f"The Studio launch file must be a regular file: {path}"
                        )
                    actual = stream.read(len(expected) + 1)
            except FileNotFoundError:
                continue
            if not managed or actual != expected:
                raise DesktopStartRefused(
                    f"Studio will not replace the existing launch file: {path}"
                )

    def register(self, *, managed: bool) -> None:
        """Atomically register owned launch files without changing the user's PATH.

        Parameters
        ----------
        managed : bool
            True only after verifying this installation's ownership record.

        Raises
        ------
        DesktopStartRefused
            A pre-existing file belongs to another application or was modified.
        OSError
            A directory, temporary file or atomic replacement cannot be written.
        """
        self.verify(managed=managed)
        for path, (content, mode) in self._files().items():
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            _write_file(path, content, mode)

    def remove(self) -> None:
        """Remove verified owned launch files while preserving every parent directory.

        Raises
        ------
        DesktopStartRefused
            A launch file was changed or replaced with a symlink.
        OSError
            Reading or removing a launch file fails.
        """
        self.verify(managed=True)
        for path in self.paths:
            path.unlink(missing_ok=True)

    def _files(self) -> dict[Path, tuple[bytes, int]]:
        home = self.profile.legacy_projects.parents[2]
        data_home = self.profile.data.parents[1]
        python = self.root / "current/runtime/bin/python3.12"
        command = home / ".local/bin/snn-studio"
        if any(character in str(path) for path in (python, command) for character in "\n\r\t"):
            raise DesktopStartRefused("The Studio installation path cannot be a desktop command.")
        launcher = "#!/bin/sh\n" + _HEADER + "exec " + shlex.quote(str(python))
        launcher += " -I -B -m sc_neurocore.studio.desktop"
        manager = launcher + '.install "$@"\n'
        launcher += ' "$@"\n'
        argument = str(command).replace("%", "%%")
        argument = "".join("\\" + ch if ch in '\\"`$' else ch for ch in argument)
        argument = '"' + argument.replace("\\", "\\\\") + '"'
        icon = str(self.root / "current/share/icons/snn-studio.svg").replace("\\", "\\\\")
        entry = _HEADER + (
            "[Desktop Entry]\nType=Application\nName=SNN Studio\n"
            "Comment=Build and explore spiking neural networks\n"
            f"Exec=/bin/sh {argument}\nIcon={icon}\n"
            "Terminal=false\nCategories=Education;Science;\nStartupNotify=false\n"
        )
        return {
            command: (launcher.encode(), 0o755),
            home / ".local/bin/snn-studio-manage": (manager.encode(), 0o755),
            data_home / "applications/li.anulum.SNNStudio.desktop": (entry.encode(), 0o644),
        }


def _write_file(path: Path, content: bytes, mode: int) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
