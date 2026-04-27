# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN datastream contract tests

"""Tests for the SCPN inter-repository datastream contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.scpn import (
    SCHEMA_VERSION,
    SCPNDatastream,
    generate_scpn_datastream,
    generate_scpn_datastream_payload,
    read_scpn_datastream,
    validate_scpn_datastream,
    write_scpn_datastream,
)


def test_generated_stream_has_canonical_shape_and_metadata() -> None:
    stream = generate_scpn_datastream(n_steps=12, dt_s=0.02, seed=11)

    assert stream.n_steps == 12
    assert stream.n_layers == 16
    assert stream.probabilities.shape == (12, 16)
    assert stream.spike_train.shape == (12, 16)
    assert stream.omega_rad_s.shape == (16,)
    assert stream.knm.shape == (16, 16)
    assert np.allclose(stream.knm, stream.knm.T)
    assert np.allclose(np.diag(stream.knm), 0.0)
    assert set(np.unique(stream.spike_train)).issubset({0, 1})


def test_rotation_angles_match_quantum_bridge_convention() -> None:
    stream = generate_scpn_datastream(n_steps=20, seed=7)
    expected = stream.spike_train.mean(axis=0) * np.pi

    np.testing.assert_allclose(stream.rotation_angles_rad, expected, atol=1e-12)
    assert np.all((stream.rotation_angles_rad >= 0.0) & (stream.rotation_angles_rad <= np.pi))


def test_quantum_amplitudes_obey_born_rule() -> None:
    stream = generate_scpn_datastream(n_steps=24, seed=9)
    amplitudes = stream.quantum_amplitudes

    assert amplitudes.shape == (16, 2)
    recovered = amplitudes[:, 1] ** 2
    np.testing.assert_allclose(recovered, stream.firing_rates, atol=1e-12)
    np.testing.assert_allclose(np.sum(amplitudes**2, axis=1), 1.0, atol=1e-12)


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


def test_validation_rejects_non_binary_spikes() -> None:
    stream = generate_scpn_datastream(n_steps=4, seed=1)
    bad = SCPNDatastream(
        dt_s=stream.dt_s,
        seed=stream.seed,
        probabilities=stream.probabilities,
        spike_train=stream.spike_train.astype(np.uint8).copy(),
        omega_rad_s=stream.omega_rad_s,
        knm=stream.knm,
    )
    bad.spike_train[0, 0] = 2

    with pytest.raises(ValueError, match="binary"):
        validate_scpn_datastream(bad)


def test_generation_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        generate_scpn_datastream(n_steps=0)

    with pytest.raises(ValueError, match="dt_s"):
        generate_scpn_datastream(dt_s=0.0)
