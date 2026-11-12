# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual validator process provenance

"""Collect interpreter facts inside the pytest process that executes validators."""

from __future__ import annotations

import getpass
import hashlib
import platform
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def facet_runtime_properties(record_testsuite_property: Callable[[str, str], None]) -> None:
    """Attach actual child runtime facts to its private JUnit report.

    Parameters
    ----------
    record_testsuite_property : Callable[[str, str], None]
        Pytest's public session property recorder. Private location properties
        are removed from the documented public projection by the parent.
    """
    executable = Path(sys.executable).resolve(strict=True)
    with executable.open("rb") as stream:
        executable_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    properties = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "executable_sha256": executable_sha256,
        "pytest": pytest.__version__,
        "executable": str(executable),
        "prefix": sys.prefix,
        "hostname": platform.node(),
        "user": getpass.getuser(),
        "home": str(Path.home()),
        "temporary": tempfile.gettempdir(),
    }
    for name, value in properties.items():
        record_testsuite_property("facet_runtime." + name, value)
