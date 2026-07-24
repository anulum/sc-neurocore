# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cli_formal.py

from __future__ import annotations


"""Exercise cli formal behaviour through the public CLI."""


import json


import subprocess


from pathlib import Path


from unittest import mock


import pytest


from sc_neurocore.formal import validate_formal_network_report


from tests.cli_test_support import run_cli


__all__ = [
    "json",
    "subprocess",
    "Path",
    "mock",
    "pytest",
    "validate_formal_network_report",
    "run_cli",
]
