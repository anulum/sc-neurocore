# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_universal_dsl.py

from __future__ import annotations

"""Test suite for the Universal Neuron DSL.

Covers:
- TOML and JSON schema loading
- Bare-name resolution against bundled schemas
- Simulation parity with hand-crafted model classes
- Parameter overrides and integration method switching
- Schema export (to_json, to_toml)
- Error handling for invalid/missing schemas
- Forward-compatible extension fields
- Schema version gating
- Introspection methods
"""
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch
from sc_neurocore.neurons.universal_dsl import (
    UniversalNeuron,
    list_bundled_schemas,
    load_schema,
    schema_to_toml,
)

__all__ = ['json', 'sys', 'Path', 'Any', 'np', 'pytest', 'MonkeyPatch', 'UniversalNeuron', 'list_bundled_schemas', 'load_schema', 'schema_to_toml']
