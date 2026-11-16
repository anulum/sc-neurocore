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


def test_compatibility_aliases_do_not_overwrite_canonical_descriptors() -> None:
    """The corpus writer visits each shared descriptor path exactly once."""
    tool = _load_tool()

    assert "KilincBhattMapNeuron" not in tool._descriptor_identities()
    assert "SCAdaptiveThresholdMapNeuron" in tool._descriptor_identities()


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


def test_schema_alias_and_event_only_integration_contracts_are_preserved() -> None:
    """Aliases and event-only schemas retain their authored integration semantics."""
    from sc_neurocore.neurons.descriptor_generator import generate_descriptor

    exp_if = generate_descriptor("ExpIFNeuron")
    mcculloch_pitts = generate_descriptor("McCullochPittsNeuron")
    poisson = generate_descriptor("PoissonNeuron")

    assert (exp_if.dt, exp_if.integration_method) == (0.02, "rk4")
    assert (mcculloch_pitts.dt, mcculloch_pitts.integration_method) == (1.0, "map")
    assert (poisson.dt, poisson.integration_method) == (1.0, "poisson_interval")


def test_hodgkin_huxley_descriptor_tracks_the_public_default_profile() -> None:
    """The separate RK4 compiler schema must not replace the Python runtime default."""
    from sc_neurocore.neurons.descriptor_generator import generate_descriptor
    from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

    descriptor = generate_descriptor("HodgkinHuxleyNeuron")

    assert HodgkinHuxleyNeuron().integrator == "baseline_euler"
    assert descriptor.integration_method == "baseline_euler"


def test_dendritic_nmda_descriptor_names_its_runtime_integrator() -> None:
    """The descriptor names the RK4 path the class runs by default, as every backend lane does.

    The model has no v1 schema, so without its structural entry the generator wrote the
    generic ``euler`` default into the descriptor.
    """
    from sc_neurocore.neurons.descriptor_generator import generate_descriptor
    from sc_neurocore.neurons.models.dendritic_nmda import DendriticNMDANeuron

    descriptor = generate_descriptor("DendriticNMDANeuron")

    assert DendriticNMDANeuron().integrator == "rk4"
    assert descriptor.integration_method == "rk4"


#: The only classes whose descriptor may name another method than their
#: ``integrator`` default, with the method it names. AdEx's explicit Euler step is
#: ``baseline_euler`` in the class and ``euler`` in its v1 schema; the descriptor
#: keeps the schema's spelling because the AdEx benchmark record binds it. A new
#: structural method entry for a class with an ``integrator`` field must be added
#: here as well, or this test fails.
RUNTIME_INTEGRATOR_EXCEPTIONS = {"AdExNeuron": "euler"}


def test_every_descriptor_names_its_class_runtime_integrator() -> None:
    """A class that declares an integrator has that integrator's default as its descriptor's method.

    The expectation comes from the class and from this test's own exception table, not from
    the generator's structural entries, so a wrong method entry added there fails here.
    """
    import dataclasses

    from sc_neurocore.neurons.descriptor_generator import _load_class, generate_descriptor
    from sc_neurocore.neurons.models import _CLASS_TO_MODULE

    checked: list[tuple[str, str, str]] = []
    for name in sorted(_CLASS_TO_MODULE):
        cls = _load_class(name)
        if not dataclasses.is_dataclass(cls):
            continue
        fields = {field.name: field.default for field in dataclasses.fields(cls)}
        default = fields.get("integrator")
        if not isinstance(default, str):
            continue
        expected = RUNTIME_INTEGRATOR_EXCEPTIONS.get(name, default)
        checked.append((name, generate_descriptor(name).integration_method, expected))

    assert len(checked) == 21
    assert [entry for entry in checked if entry[1] != entry[2]] == []


def test_public_rng_state_properties_are_materialised() -> None:
    """Seeded stochastic descriptors expose replay state from their public property."""
    from sc_neurocore.neurons.descriptor_generator import generate_descriptor

    for model in ("EscapeRateNeuron", "PoissonNeuron"):
        state = {item.name: item.init for item in generate_descriptor(model).state}
        assert state["rng_state"] == 44257.0


def test_a_refresh_keeps_a_curated_case_that_sets_parameters() -> None:
    """A curated behaviour case keeps its ``[behaviour.parameters]`` through a corpus refresh.

    The Mihalas-Niebur cases set both current jumps, which the class leaves at 0, so that
    the jump terms act. A refresh regenerates the structural tables from the model code and
    merges the committed curation back. The cases must come out as committed, their
    parameters included, and the descriptor parser must read the same overrides from the
    refreshed text.
    """
    import tomllib

    from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
    from sc_neurocore.neurons.model_descriptor import parse_model_descriptor

    tool = _load_tool()
    committed = load_descriptor_payload("MihalasNieburNeuron")
    assert committed is not None
    with_parameters = [case for case in committed.get("behaviour", []) if case.get("parameters")]
    assert with_parameters, "the Mihalas-Niebur descriptor states no case that sets parameters"

    refreshed = tomllib.loads(tool._rendered_descriptor("MihalasNieburNeuron"))

    assert refreshed["behaviour"] == committed["behaviour"]
    read_back = parse_model_descriptor(refreshed).behaviour
    assert [dict(case.parameters) for case in read_back if case.parameters] == [
        case["parameters"] for case in with_parameters
    ]
