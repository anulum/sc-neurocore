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

import sc_neurocore.scpn.datastream as datastream_module
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


def test_from_json_rejects_unknown_schema() -> None:
    payload = generate_scpn_datastream_payload(n_steps=3, seed=2)
    payload["schema_version"] = "sc-neurocore.scpn.datastream.v0"

    with pytest.raises(ValueError, match="unsupported"):
        SCPNDatastream.from_json_dict(payload)


def test_validation_rejects_shape_and_bound_violations() -> None:
    stream = generate_scpn_datastream(n_steps=4, seed=3)

    cases = [
        (
            "matching shapes",
            dict(probabilities=stream.probabilities[:-1]),
        ),
        (
            "2-D",
            dict(
                probabilities=stream.probabilities.reshape(-1),
                spike_train=stream.spike_train.reshape(-1),
            ),
        ),
        (
            "layer columns",
            dict(
                probabilities=stream.probabilities[:, :-1], spike_train=stream.spike_train[:, :-1]
            ),
        ),
        (
            "omega_rad_s",
            dict(omega_rad_s=stream.omega_rad_s[:-1]),
        ),
        (
            "knm must have shape",
            dict(knm=stream.knm[:-1, :]),
        ),
        (
            "probabilities must be in",
            dict(probabilities=stream.probabilities.copy()),
        ),
        (
            "knm must be symmetric",
            dict(knm=stream.knm.copy()),
        ),
        (
            "knm diagonal",
            dict(knm=stream.knm.copy()),
        ),
    ]

    cases[5][1]["probabilities"][0, 0] = 1.1
    cases[6][1]["knm"][0, 1] += 0.5
    cases[7][1]["knm"][0, 0] = 0.5

    for match, overrides in cases:
        bad = SCPNDatastream(
            dt_s=overrides.get("dt_s", stream.dt_s),
            seed=stream.seed,
            probabilities=overrides.get("probabilities", stream.probabilities),
            spike_train=overrides.get("spike_train", stream.spike_train),
            omega_rad_s=overrides.get("omega_rad_s", stream.omega_rad_s),
            knm=overrides.get("knm", stream.knm),
        )
        with pytest.raises(ValueError, match=match):
            validate_scpn_datastream(bad)


def test_generation_rejects_invalid_probability_bounds() -> None:
    with pytest.raises(ValueError, match="spike_floor"):
        generate_scpn_datastream(spike_floor=0.5, spike_ceiling=0.5)

    with pytest.raises(ValueError, match="spike_floor"):
        generate_scpn_datastream(spike_floor=-0.1, spike_ceiling=0.9)

    with pytest.raises(ValueError, match="spike_floor"):
        generate_scpn_datastream(spike_floor=0.1, spike_ceiling=1.1)


def test_generation_handles_zero_coupling_span(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datastream_module, "build_knm_matrix", lambda: np.zeros((16, 16)))

    stream = generate_scpn_datastream(n_steps=3, seed=4)

    np.testing.assert_allclose(stream.knm, 0.0)
    assert stream.probabilities.shape == (3, 16)


def test_validation_rejects_non_positive_dt() -> None:
    stream = generate_scpn_datastream(n_steps=3, seed=6)
    bad = SCPNDatastream(
        dt_s=0.0,
        seed=stream.seed,
        probabilities=stream.probabilities,
        spike_train=stream.spike_train,
        omega_rad_s=stream.omega_rad_s,
        knm=stream.knm,
    )

    with pytest.raises(ValueError, match="dt_s"):
        validate_scpn_datastream(bad)


def test_read_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "bad_stream.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="root"):
        read_scpn_datastream(path)


def test_generation_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        generate_scpn_datastream(n_steps=0)

    with pytest.raises(ValueError, match="dt_s"):
        generate_scpn_datastream(dt_s=0.0)


def test_validation_rejects_non_finite_arrays() -> None:
    stream = generate_scpn_datastream(n_steps=4, seed=5)
    base = dict(
        dt_s=stream.dt_s,
        seed=stream.seed,
        probabilities=stream.probabilities,
        spike_train=stream.spike_train,
        omega_rad_s=stream.omega_rad_s,
        knm=stream.knm,
    )

    probs = stream.probabilities.copy()
    probs[0, 0] = np.inf
    with pytest.raises(ValueError, match="probabilities must be finite"):
        validate_scpn_datastream(SCPNDatastream(**{**base, "probabilities": probs}))

    omega = stream.omega_rad_s.copy()
    omega[0] = np.inf
    with pytest.raises(ValueError, match="omega_rad_s must be finite"):
        validate_scpn_datastream(SCPNDatastream(**{**base, "omega_rad_s": omega}))

    knm = stream.knm.copy()
    knm[0, 1] = np.inf
    knm[1, 0] = np.inf
    with pytest.raises(ValueError, match="knm must be finite"):
        validate_scpn_datastream(SCPNDatastream(**{**base, "knm": knm}))


def test_numeric_array_from_payload_rejects_ragged_values() -> None:
    with pytest.raises(ValueError, match="must be a numeric JSON array"):
        datastream_module._numeric_array_from_payload([[1.0, 2.0], [3.0]], key="knm")
