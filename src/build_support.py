# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Build-owned model documentation resources

"""Package canonical model pages without keeping a second editable source copy."""

from pathlib import Path

from setuptools.command.build_py import build_py


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
