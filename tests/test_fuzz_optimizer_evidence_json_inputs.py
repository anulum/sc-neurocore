# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for optimiser evidence JSON

"""Property-based fuzz tests for optimiser synthesis-evidence JSON inputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.optimizer import load_observations, observations_from_payload
from sc_neurocore.optimizer.observation_loader import ObservationLoadError

_INT_FIELDS = st.sampled_from(
    ("mac_count", "bitstream_length", "precision_bits", "luts_used", "latency_cycles")
)
_FLOAT_FIELDS = st.sampled_from(("power_mw", "accuracy_score"))
_BAD_INT_VALUE = (
    st.booleans()
    | st.floats(allow_nan=True, allow_infinity=True).filter(
        lambda value: not math.isfinite(value) or not value.is_integer()
    )
    | st.integers(max_value=-1)
    | st.text(min_size=1, max_size=16).filter(lambda value: not _is_non_negative_int_text(value))
)
_BAD_FLOAT_VALUE = (
    st.booleans()
    | st.sampled_from([float("nan"), float("inf"), float("-inf")])
    | st.floats(max_value=-1.0, allow_nan=False, allow_infinity=False)
    | st.text(min_size=1, max_size=16).filter(lambda value: not _is_non_negative_float_text(value))
)


def _valid_record() -> dict[str, Any]:
    return {
        "mac_count": 256,
        "bitstream_length": 128,
        "decorrelator": "LFSR",
        "mode": "SC",
        "precision_bits": 8,
        "lfsr_polynomial": "x16+x15+x13+x4+1",
        "luts_used": 320,
        "power_mw": 1.5,
        "latency_cycles": 128,
        "accuracy_score": 0.997,
    }


def _is_non_negative_int_text(value: str) -> bool:
    try:
        parsed = int(value)
    except ValueError:
        return False
    return parsed >= 0


def _is_non_negative_float_text(value: str) -> bool:
    try:
        parsed = float(value)
    except ValueError:
        return False
    return math.isfinite(parsed) and parsed >= 0.0


@given(field=_INT_FIELDS, value=_BAD_INT_VALUE)
@settings(max_examples=160, deadline=None)
def test_fuzz_observation_loader_rejects_invalid_integer_metrics(field: str, value: object) -> None:
    record = _valid_record()
    record[field] = value

    with pytest.raises(ObservationLoadError, match=f"{field} must be"):
        observations_from_payload({"observations": [record]}, source="fuzz.json")


@given(field=_FLOAT_FIELDS, value=_BAD_FLOAT_VALUE)
@settings(max_examples=160, deadline=None)
def test_fuzz_observation_loader_rejects_invalid_float_metrics(field: str, value: object) -> None:
    record = _valid_record()
    record[field] = value

    with pytest.raises(ObservationLoadError, match=f"{field} must be"):
        observations_from_payload({"observations": [record]}, source="fuzz.json")


@given(
    payload=st.recursive(st.none() | st.booleans() | st.text(max_size=16), st.lists, max_leaves=8)
)
@settings(max_examples=80, deadline=None)
def test_fuzz_observation_loader_rejects_non_mapping_records(payload: object) -> None:
    if isinstance(payload, dict):
        return

    with pytest.raises(ObservationLoadError, match="record must be a JSON object"):
        observations_from_payload({"observations": [payload]}, source="fuzz.json")


def test_load_observations_rejects_json_nan_metric(tmp_path: Path) -> None:
    record = _valid_record()
    record["power_mw"] = float("nan")
    path = tmp_path / "nan_observation.json"
    path.write_text(json.dumps({"observations": [record]}), encoding="utf-8")

    with pytest.raises(ObservationLoadError, match="power_mw must be finite"):
        load_observations(path)
