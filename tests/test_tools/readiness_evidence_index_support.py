# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_readiness_evidence_index.py

from __future__ import annotations

"""Real-surface tests for tools/readiness_evidence_index.py.

Exercises the shipped module (not a reimplementation): inventory construction,
facet builders, apply dry-path against a temporary descriptor payload, and the
CLI entry points."""

import importlib.util

import json

import sys

from pathlib import Path

from types import ModuleType

from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import pytest

import tomli_w

from sc_neurocore.neurons.model_descriptor import ModelDescriptor

REPO_ROOT = Path(__file__).resolve().parents[2]

TOOL_PATH = REPO_ROOT / "tools" / "readiness_evidence_index.py"


def _load_tool() -> ModuleType:
    """Load the readiness evidence index tool as a real module from disk."""
    import sys

    name = "readiness_evidence_index_under_test"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses with slots require the module to be registered first.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    """Shared loaded tool module."""
    return _load_tool()


__all__ = [
    "importlib",
    "json",
    "sys",
    "Path",
    "ModuleType",
    "Any",
    "tomllib",
    "pytest",
    "tomli_w",
    "ModelDescriptor",
    "REPO_ROOT",
    "TOOL_PATH",
    "_load_tool",
    "tool",
]
