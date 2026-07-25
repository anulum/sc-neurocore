# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN datastream JSON contracts

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sc_neurocore.scpn.datastream as datastream_module
from sc_neurocore.scpn import (
    SCHEMA_VERSION,
    SCPNDatastream,
    generate_scpn_datastream,
    generate_scpn_datastream_payload,
    read_scpn_datastream,
    write_scpn_datastream,
)


def test_json_payload_roundtrip(tmp_path: Path) -> None:
    stream = generate_scpn_datastream(n_steps=8, dt_s=0.005, seed=123)
    path = tmp_path / "scpn_stream.json"

    write_scpn_datastream(path, stream)
    loaded = read_scpn_datastream(path)

    assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION
    assert loaded.dt_s == stream.dt_s
    assert loaded.seed == stream.seed
    np.testing.assert_allclose(loaded.probabilities, stream.probabilities)
    np.testing.assert_array_equal(loaded.spike_train, stream.spike_train)
    np.testing.assert_allclose(loaded.rotation_angles_rad, stream.rotation_angles_rad)


def test_payload_helper_is_json_friendly() -> None:
    payload = generate_scpn_datastream_payload(n_steps=5, seed=5)

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["source_project"] == "sc-neurocore"
    assert len(payload["spike_train"]) == 5
    assert len(payload["spike_train"][0]) == 16
    json.dumps(payload)


def test_from_json_rejects_unknown_schema() -> None:
    payload = generate_scpn_datastream_payload(n_steps=3, seed=2)
    payload["schema_version"] = "sc-neurocore.scpn.datastream.v0"

    with pytest.raises(ValueError, match="unsupported"):
        SCPNDatastream.from_json_dict(payload)


def test_read_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "bad_stream.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="root"):
        read_scpn_datastream(path)


def test_numeric_array_from_payload_rejects_ragged_values() -> None:
    with pytest.raises(ValueError, match="must be a numeric JSON array"):
        datastream_module._numeric_array_from_payload([[1.0, 2.0], [3.0]], key="knm")
