# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for chip-spec JSON

"""Property-based fuzz tests for neuromorphic chip-spec JSON inputs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.chip_compiler import ChipSpec, load_chip_spec

_TOP_LEVEL_FIELDS = ("name", "vendor", "total_cores", "core")
_CORE_FIELDS = (
    "max_neurons",
    "max_synapses_per_neuron",
    "weight_bits",
    "supported_neuron_types",
)
_JSON_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-8, max_value=8)
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(max_size=12)
)
_JSON_VALUE = st.recursive(
    _JSON_SCALAR,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=24,
)
_EXTRA_KEY = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{0,10}", fullmatch=True).filter(
    lambda key: (
        key
        not in {
            "name",
            "vendor",
            "total_cores",
            "core",
            "clock_mhz",
            "power_mw_per_core",
            "routing_topology",
            "max_fan_out",
            "analog_noise_cv",
            "max_neurons",
            "max_synapses_per_neuron",
            "weight_bits",
            "supported_neuron_types",
            "has_on_chip_learning",
            "learning_rules",
            "max_delay_steps",
        }
    )
)


def _valid_spec_payload() -> dict[str, Any]:
    return {
        "name": "test_chip",
        "vendor": "Anulum",
        "total_cores": 2,
        "clock_mhz": 125.0,
        "power_mw_per_core": 0.25,
        "routing_topology": "mesh",
        "max_fan_out": 128,
        "analog_noise_cv": 0.0,
        "core": {
            "max_neurons": 64,
            "max_synapses_per_neuron": 512,
            "weight_bits": 8,
            "supported_neuron_types": ["LIF"],
            "has_on_chip_learning": False,
            "learning_rules": ["STDP"],
            "max_delay_steps": 4,
        },
    }


def _load_payload(payload: object) -> ChipSpec:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "chip.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_chip_spec(path)


def test_load_chip_spec_roundtrip() -> None:
    chip = _load_payload(_valid_spec_payload())

    assert chip.name == "test_chip"
    assert chip.total_neurons == 128
    assert chip.core.weight_bits == 8
    assert chip.routing_topology == "mesh"


@given(payload=_JSON_VALUE)
@settings(max_examples=120, deadline=None)
def test_fuzz_load_chip_spec_rejects_malformed_payloads(payload: object) -> None:
    try:
        chip = _load_payload(payload)
    except ValueError:
        return

    assert chip.total_cores > 0
    assert chip.core.max_neurons > 0
    assert chip.core.supported_neuron_types


@given(missing=st.sampled_from(_TOP_LEVEL_FIELDS))
@settings(max_examples=24, deadline=None)
def test_fuzz_load_chip_spec_rejects_missing_top_level_fields(missing: str) -> None:
    payload = _valid_spec_payload()
    del payload[missing]

    with pytest.raises(ValueError, match="missing"):
        _load_payload(payload)


@given(missing=st.sampled_from(_CORE_FIELDS))
@settings(max_examples=24, deadline=None)
def test_fuzz_load_chip_spec_rejects_missing_core_fields(missing: str) -> None:
    payload = _valid_spec_payload()
    del payload["core"][missing]

    with pytest.raises(ValueError, match="missing"):
        _load_payload(payload)


@given(extra_key=_EXTRA_KEY)
@settings(max_examples=80, deadline=None)
def test_fuzz_load_chip_spec_rejects_unexpected_top_level_fields(extra_key: str) -> None:
    payload = _valid_spec_payload()
    payload[extra_key] = 1

    with pytest.raises(ValueError, match="unexpected"):
        _load_payload(payload)


@given(extra_key=_EXTRA_KEY)
@settings(max_examples=80, deadline=None)
def test_fuzz_load_chip_spec_rejects_unexpected_core_fields(extra_key: str) -> None:
    payload = _valid_spec_payload()
    payload["core"][extra_key] = 1

    with pytest.raises(ValueError, match="unexpected"):
        _load_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "", "non-empty string"),
        ("vendor", False, "non-empty string"),
        ("total_cores", 0, "positive"),
        ("clock_mhz", -1.0, "positive"),
        ("power_mw_per_core", -0.1, "non-negative"),
        ("routing_topology", "torus", "routing_topology"),
        ("max_fan_out", -1, "non-negative"),
        ("analog_noise_cv", float("inf"), "finite"),
    ],
)
def test_load_chip_spec_rejects_bad_top_level_values(
    field: str, value: object, message: str
) -> None:
    payload = _valid_spec_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        _load_payload(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_neurons", 0, "positive"),
        ("max_synapses_per_neuron", -1, "positive"),
        ("weight_bits", True, "integer"),
        ("supported_neuron_types", [], "non-empty list of strings"),
        ("has_on_chip_learning", "false", "boolean"),
        ("learning_rules", [""], "non-empty list of strings"),
        ("max_delay_steps", -1, "non-negative"),
    ],
)
def test_load_chip_spec_rejects_bad_core_values(field: str, value: object, message: str) -> None:
    payload = _valid_spec_payload()
    payload["core"][field] = value

    with pytest.raises(ValueError, match=message):
        _load_payload(payload)
