# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public native equation source replay

"""Replay complete source contracts through the public compiler and native parser."""

from __future__ import annotations

import hashlib
import json

import pytest

from sc_neurocore.compiler import compile_to_ir, recompile_ir
from sc_neurocore.neurons.equation_builder import EquationNeuron
from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import parse_ir


@pytest.mark.parametrize("case", ["ordered", "random", "macro", "pipeline"])
def test_public_equation_replay_preserves_source_and_physical_cadence(case: str) -> None:
    """Reconstruction retains initial/reset order, seed sites and actual register periods."""
    if case == "ordered":
        neuron = EquationNeuron(
            {"w": "v - w", "v": "w + I"},
            state={"w": 0.5, "v": -0.25},
            threshold="v > 1",
            reset={"v": "w", "w": "v_prev"},
            spike_value="v - w_prev",
            method="gauss_seidel",
            dt=0.5,
        )
    elif case == "random":
        neuron = EquationNeuron(
            {"v": "rand() + I"},
            method="map",
            dt=0.5,
            threshold="v > 1",
            reset={"v": "0"},
            rng_seed=(1 << 63) + 3,
            random_stream=(1 << 31) + 5,
        )
    elif case == "macro":
        neuron = EquationNeuron(
            {"v": "I"},
            threshold="v > v_prev",
            spike_value="v - v_prev",
            detection="crossing",
            substeps=3,
            method="map",
            dt=0.5,
        )
    else:
        neuron = EquationNeuron({"v": "gain * (I - v)"}, parameters={"gain": 0.5}, dt=0.5)
    original = compile_to_ir(
        neuron,
        "replay_subject",
        rounding="nearest",
        overflow="wrap",
        pipeline_points=["mul0"] if case == "pipeline" else [],
        random_precision="exact",
        function_lowering="v2",
    )
    restored = recompile_ir(original.graph.to_text())
    assert restored.graph.to_text() == original.graph.to_text()
    assert restored.graph.emit_sv() == original.graph.emit_sv()
    assert restored.graph.emit_mlir() == original.graph.emit_mlir()
    assert restored.output_widths == original.output_widths
    assert restored.pipeline_registers == original.pipeline_registers
    assert restored.clocks_per_timestep == (
        3 if case == "macro" else 2 if case == "pipeline" else 1
    )


@pytest.mark.parametrize("with_input", [True, False])
def test_public_equation_replay_retains_strict_base_and_display_units(with_input: bool) -> None:
    """Strict source replay repeats dimensional validation while keeping display metadata."""
    from sc_neurocore.neurons._units import UNIT_REGISTRY, require_pint

    require_pint()
    neuron = EquationNeuron(
        {"v": "(I - v) / tau" if with_input else "(bias - v) / tau"},
        state={"v": 1000.0 * UNIT_REGISTRY.millivolt},
        parameters={"tau": 2.0 * UNIT_REGISTRY.second},
        constants={} if with_input else {"bias": 0.25 * UNIT_REGISTRY.volt},
        dt=0.5 * UNIT_REGISTRY.second,
        units="strict",
        input_unit=1.0 * UNIT_REGISTRY.volt if with_input else None,
    )
    original = compile_to_ir(neuron)
    restored = recompile_ir(original.graph.to_text())
    assert original.graph.to_text() == restored.graph.to_text()
    payload = original.graph.source_payload
    assert payload is not None
    model = json.loads(payload)["model"]
    assert model["initial_state"] == [["v", 1.0]]
    assert model["units"] == "strict"
    assert model["display_state_units"] == {"v": "millivolt"}
    assert model["base_state_units"]["v"] != model["display_state_units"]["v"]


def _payload(text: str, payload: str) -> str:
    graph = parse_ir(text)
    previous = graph.source_payload
    assert previous is not None
    source = graph.source_sha256
    assert source is not None
    changed = text.replace(
        "source=" + source, "source=" + hashlib.sha256(payload.encode()).hexdigest()
    )
    changed = changed.replace(
        "payload=" + previous.encode().hex(), "payload=" + payload.encode().hex()
    )
    assert changed != text
    assert parse_ir(changed).verify() is None
    return changed


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("root", "schema", "unknown"),
        ("root", "model", []),
        ("root", "extra", 1),
        ("numeric", "signed", 1),
        ("numeric", "fraction", True),
        ("numeric", "rounding", []),
        ("numeric", "data_width", 0),
        ("pipeline", "points", {}),
        ("pipeline", "points", [False]),
        ("pipeline", "stages", True),
        ("model", "dt", True),
        ("model", "dt", float("nan")),
        ("model", "threshold", []),
        ("model", "parameters", {}),
        ("model", "parameters", [["tau"]]),
        ("model", "parameters", [["tau", "2"]]),
        ("model", "parameters", [["tau", 2], ["tau", 2]]),
        ("model", "runtime_units", []),
        ("model", "constants", [[2, 1]]),
    ],
)
def test_public_equation_replay_refuses_noncanonical_or_malformed_source(
    section: str, field: str, value: object
) -> None:
    """Native-valid checksums do not conceal a malformed or unsupported source schema."""
    original = compile_to_ir(EquationNeuron({"v": "I / tau"}, parameters={"tau": 2.0}, dt=0.5))
    payload = original.graph.source_payload
    assert payload is not None
    raw = json.loads(payload)
    target = raw if section == "root" else raw[section]
    target[field] = value
    altered = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError):
        recompile_ir(_payload(original.graph.to_text(), altered))


def test_public_equation_replay_checks_model_graph_consistency() -> None:
    """A validly rehashed different equation still fails the actual recompilation comparison."""
    original = compile_to_ir(EquationNeuron({"v": "I - v"}, dt=0.5))
    payload = original.graph.source_payload
    assert payload is not None
    raw = json.loads(payload)
    raw["model"]["equations"] = [["v", "I + v"]]
    altered = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="does not match native graph"):
        recompile_ir(_payload(original.graph.to_text(), altered))


def test_public_equation_replay_requires_model_source_and_valid_json() -> None:
    """A source-free real model and a rehashed malformed payload have distinct authored refusals."""
    original = compile_to_ir(EquationNeuron({"v": "I - v"}, dt=0.5))
    text = original.graph.to_text()
    plain = "\n".join(line for line in text.splitlines() if "sc.source " not in line) + "\n"
    assert parse_ir(plain).verify() is None
    with pytest.raises(ValueError, match="no equation source contract"):
        recompile_ir(plain)
    with pytest.raises(ValueError, match="invalid equation source contract"):
        recompile_ir(_payload(text, "{"))


@pytest.mark.parametrize(
    "display_units",
    [
        {"v": "kilogram"},
        {"v": "millivolt", "ghost": "volt"},
        {},
    ],
    ids=["incompatible-dimension", "unknown-state", "missing-state"],
)
def test_public_equation_replay_refuses_invalid_display_units(
    display_units: dict[str, str],
) -> None:
    """Require declared state names and physical dimensions despite valid native checksums.

    Parameters
    ----------
    display_units : dict of str to str
        Changed strict-unit presentation map in an otherwise valid source contract.
    """
    from sc_neurocore.neurons._units import UNIT_REGISTRY, require_pint

    require_pint()
    neuron = EquationNeuron(
        {"v": "(I - v) / tau"},
        state={"v": 1000.0 * UNIT_REGISTRY.millivolt},
        parameters={"tau": 2.0 * UNIT_REGISTRY.second},
        dt=0.5 * UNIT_REGISTRY.second,
        units="strict",
        input_unit=1.0 * UNIT_REGISTRY.volt,
    )
    original = compile_to_ir(neuron)
    payload = original.graph.source_payload
    assert payload is not None
    raw = json.loads(payload)
    raw["model"]["display_state_units"] = display_units
    altered = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="display units"):
        recompile_ir(_payload(original.graph.to_text(), altered))
