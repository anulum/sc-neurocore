# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_adapter_ecosystem.py

from __future__ import annotations

import importlib
from pathlib import Path
import sys
from textwrap import dedent
from typing import cast
import numpy as np
import pytest
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
REPO_ROOT = Path(__file__).resolve().parents[1]
def _pyproject_adapter_entry_points(group: str) -> dict[str, str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = pyproject["project"]["entry-points"][group]
    assert isinstance(entry_points, dict)
    result: dict[str, str] = {}
    for name, target in entry_points.items():
        assert isinstance(name, str)
        assert isinstance(target, str)
        result[name] = target
    return result
def _resolve_entry_point_target(target: str) -> type:
    module_name, separator, attribute_path = target.partition(":")
    assert module_name
    assert separator == ":"
    assert attribute_path
    resolved: object = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        resolved = getattr(resolved, attribute)
    assert isinstance(resolved, type)
    return resolved

__all__ = ['importlib', 'Path', 'sys', 'dedent', 'cast', 'np', 'pytest', 'REPO_ROOT', '_pyproject_adapter_entry_points', '_resolve_entry_point_target']
