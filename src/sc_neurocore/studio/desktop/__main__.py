# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop application entry point

"""Launch the real Studio window using this interpreter's package and runtime."""

from __future__ import annotations

import logging
import sys

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.desktop.runtime import DesktopBackend


def main() -> int:
    """Open the application and keep Qt running until owned backend cleanup finishes.

    Returns
    -------
    int
        Qt's exit status, or one when the runtime/profile cannot be opened.
    """
    try:
        from PySide6.QtWidgets import QApplication
        from sc_neurocore.studio.desktop.window import StudioWindow
    except ImportError:
        print("The Studio desktop runtime is missing Qt WebEngine.", file=sys.stderr)
        return 1
    try:
        backend = DesktopBackend()
        backend.paths.prepare()
    except AuthoredRefusal as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError:
        print(
            "Studio could not prepare its desktop directories. Check their permissions.",
            file=sys.stderr,
        )
        return 1
    logging.basicConfig(filename=backend.paths.state / "desktop.log", level=logging.WARNING)
    application = QApplication(sys.argv)
    application.setApplicationName("SNN Studio")
    application.setOrganizationDomain("anulum.li")
    window = StudioWindow(backend)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
