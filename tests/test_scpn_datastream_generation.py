# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN datastream generation contracts

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore.scpn.datastream as datastream_module
from sc_neurocore.scpn import generate_scpn_datastream


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


def test_generation_rejects_invalid_window() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        generate_scpn_datastream(n_steps=0)

    with pytest.raises(ValueError, match="dt_s"):
        generate_scpn_datastream(dt_s=0.0)
