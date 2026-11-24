# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Fixture of the former tests/gotm_brain_support.py

"""Provide the fake GOTM repository fixture of the GOTM brain and content-indexer suites.

Registered for the whole test session by ``pytest_plugins`` in ``tests/conftest.py``. The fixture name is
unique across ``tests/`` and the installed plugins, it is not autouse, and it keeps its function scope. It
imports nothing beyond pytest at run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal fake GOTM repository for testing."""
    src = tmp_path / "src" / "example"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text('"""Example module docstring."""\n')
    (src / "core.py").write_text(
        '"""Core module.\n\n'
        "Implements the main algorithm.\n"
        '"""\n\n'
        "import numpy as np\n\n"
        "def compute(x):\n"
        '    """Compute the result."""\n'
        "    return x * 2\n"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(
        "# Example Repository\n\n"
        "This is a test repository for the GOTM brain.\n\n"
        "## Features\n\n"
        "- Feature 1: quantum coupling\n"
        "- Feature 2: metabolic feedback\n"
    )
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "example"\nversion = "0.1.0"\n')
    rust_src = tmp_path / "src" / "lib.rs"
    rust_src.write_text(
        "/// Main computation function.\n"
        "/// Returns the doubled input.\n"
        "pub fn compute(x: f64) -> f64 {\n"
        "    x * 2.0\n"
        "}\n"
    )
    # Add a __pycache__ dir that should be skipped
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "junk.pyc").write_bytes(b"\x00" * 100)
    return tmp_path
