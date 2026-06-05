# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# SC-NeuroCore Sphinx configuration

import os
import sys

sys.path.insert(0, os.path.abspath("../../../src"))

project = "SC-NeuroCore"
copyright = "1998-2026, Miroslav Sotek"
author = "Miroslav Sotek"
release = "3.15.8"

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

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_title = "SC-NeuroCore API"
