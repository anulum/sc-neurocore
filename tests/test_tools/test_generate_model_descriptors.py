# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — model descriptor generator tests

"""Tests for the model descriptor corpus generator.

The generator renders every model's ``v2`` descriptor TOML. It must emit the SPDX
provenance header (so regenerated descriptors stay licence-compliant) and the committed
corpus must stay in sync with the model code.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/generate_model_descriptors.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_model_descriptors", TOOL)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rendered_descriptor_carries_spdx_header() -> None:
    """Every rendered descriptor must begin with the seven-line SPDX provenance header."""
    tool = _load_tool()

    rendered = tool._rendered_descriptor("HodgkinHuxleyNeuron")

    lines = rendered.splitlines()
    assert lines[0] == "# SPDX-License-Identifier: AGPL-3.0-or-later"
    assert lines[1] == "# Commercial license available"
    assert lines[6] == "# SC-NeuroCore — Source/config provenance header"
    assert lines[7] == ""  # blank line separates the header from the TOML body
    assert lines[8] == "[metadata]"


def test_committed_descriptor_corpus_is_in_sync() -> None:
    """The committed descriptor corpus must match the generator output exactly."""
    tool = _load_tool()

    assert tool.check_corpus() == []


def test_header_constant_matches_a_committed_descriptor() -> None:
    """The header constant must reproduce a committed descriptor's header byte-for-byte."""
    tool = _load_tool()

    committed = (
        REPO / "src/sc_neurocore/neurons/model_descriptors/HodgkinHuxleyNeuron.toml"
    ).read_text(encoding="utf-8")
    assert committed.startswith(tool._DESCRIPTOR_HEADER)


def test_map_descriptor_uses_schema_iteration_dt_without_public_dt_parameter() -> None:
    """A pure map inherits its unit iteration from the schema, not the ODE fallback."""
    from sc_neurocore.neurons.descriptor_generator import generate_descriptor
    from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron

    descriptor = generate_descriptor("RulkovMapNeuron")

    assert "dt" not in inspect.signature(RulkovMapNeuron).parameters
    assert descriptor.dt == 1.0
    assert descriptor.integration_method == "map"
