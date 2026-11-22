# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Package version source

"""The package version, in one module that imports nothing from the package.

Code that records which SC-NeuroCore produced a result reads :data:`VERSION`
from here instead of from the package root, whose import runs the whole facade.
``sc_neurocore.__version__`` re-exports it, and
``tools/version_surface_audit.py`` holds it equal to ``pyproject.toml``.
"""

from __future__ import annotations

from typing import Final

#: The release version of this source tree.
VERSION: Final[str] = "4.2.0"
