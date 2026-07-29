# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore Sphinx configuration

import os
import sys

sys.path.insert(0, os.path.abspath("../../../src"))

from sc_neurocore import __version__

project = "SC-NeuroCore"
copyright = "1998-2026, Miroslav Sotek"
author = "Miroslav Sotek"
release = __version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_mock_imports = ["nir", "cupy", "jax", "jaxlib", "qiskit", "pennylane", "numba", "torch"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = []
html_title = "SC-NeuroCore API"
