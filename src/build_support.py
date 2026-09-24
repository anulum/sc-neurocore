# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Build-owned model documentation resources

"""Package canonical model pages and the built Studio UI without a second source copy."""

import os
import shutil
from pathlib import Path

from setuptools.command.build_py import build_py

#: ``required`` makes a build without the built Studio UI fail; release builds
#: set it. Any other value packages the UI when it has been built and omits it
#: otherwise, so a source install without Node.js still works.
STUDIO_UI_REQUIREMENT = "SC_NEUROCORE_STUDIO_UI"


class BuildStudioResources(build_py):
    """Copy model reference pages into the wheel's Studio resource directory.

    The source distribution retains ``docs/api/models`` via MANIFEST.in, so
    both direct wheel builds and wheels built from an sdist use identical pages.
    Missing source documentation fails the build rather than producing a wheel
    whose Studio can only serve documentation from a neighbouring checkout.
    """

    def run(self) -> None:
        """Build Python packages, then copy each canonical Markdown page.

        Raises
        ------
        FileNotFoundError
            If the source tree contains no model reference pages.
        OSError
            If a page cannot be read or its build destination cannot be written.
        """
        source = Path(__file__).resolve().parents[1] / "docs" / "api" / "models"
        pages = sorted(source.glob("*.md"))
        if not pages:
            raise FileNotFoundError(f"No model reference pages in {source}")
        super().run()
        destination = Path(self.build_lib) / "sc_neurocore" / "studio" / "model_docs"
        self.mkpath(str(destination))
        page_names = {page.name for page in pages}
        for stale in destination.glob("*.md"):
            if stale.name not in page_names:
                self.execute(stale.unlink, ())
        self.force = True
        for page in pages:
            self.copy_file(str(page), str(destination / page.name))
        self._package_studio_ui()

    def _package_studio_ui(self) -> None:
        """Copy the built Studio frontend into the package, or refuse a release without it.

        Raises
        ------
        FileNotFoundError
            If ``SC_NEUROCORE_STUDIO_UI=required`` and the UI has not been built.
        """
        built = Path(__file__).resolve().parents[1] / "studio" / "frontend" / "dist"
        destination = Path(self.build_lib) / "sc_neurocore" / "studio" / "frontend_dist"
        if destination.exists():
            shutil.rmtree(destination)
        if not (built / "index.html").is_file():
            if os.environ.get(STUDIO_UI_REQUIREMENT) == "required":
                raise FileNotFoundError(
                    f"No built Studio UI in {built}; build it first "
                    "(cd studio/frontend && npm ci && npm run build)"
                )
            self.announce("Studio UI not built; the package carries no user interface", level=3)
            return
        shutil.copytree(built, destination)
