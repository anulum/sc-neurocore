# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN datastream bridge tests

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.optimizer.surrogate_sc_optimizer import BenchmarkObservation
from scpneurocore import (
    SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION,
    DatastreamValidationError,
    build_datastream_packet,
    load_power_grid,
    validate_datastream_payload,
)


def _waveform() -> np.ndarray:
    data = np.zeros((8, 4), dtype=np.float32)
    data[1, 0] = -8.0
    data[3, 2] = -9.0
    data[6, 1] = -7.0
    return data


def _spikes() -> np.ndarray:
    raster = np.zeros((8, 4), dtype=np.int8)
    raster[1, 0] = 1
    raster[3, 2] = 1
    raster[6, 1] = 1
    return raster


def _observation() -> BenchmarkObservation:
    return BenchmarkObservation(
        mac_count=128,
        bitstream_length=64,
        decorrelator="LFSR",
        mode="SC",
        precision_bits=8,
        lfsr_polynomial="x16+x15+x13+x4+1",
        luts_used=320,
        power_mw=7.5,
        latency_cycles=64,
        accuracy_score=0.998,
        is_critical_path=True,
    )


def test_datastream_packet_combines_waveform_aer_telemetry_and_hashes() -> None:
    qpu_artifact = load_power_grid(4, source_mode="fixture")
    packet = build_datastream_packet(
        waveform=_waveform(),
        spike_raster=_spikes(),
        source_name="unit-replay",
        source_mode="fixture",
        layer_id="sensory",
        qpu_artifact=qpu_artifact,
        optimiser_observation=_observation(),
        metadata={"window": "unit"},
    )
    payload = packet.to_bridge_dict()

    validate_datastream_payload(payload)
    assert payload["schema_version"] == SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION
    assert payload["waveform_shape"] == [8, 4]
    assert payload["spike_shape"] == [8, 4]
    assert payload["aer_metrics"]["n_spikes"] == 3
    assert payload["telemetry"]["total_ticks"] == 8
    assert payload["telemetry"]["total_spikes"] == 3
    assert payload["telemetry"]["layers"]["sensory"]["spike_count"] == 3
    assert payload["qpu_artifact_sha256"] == qpu_artifact.artifact_sha256
    assert payload["optimiser_observation"]["luts_used"] == 320
    assert (
        packet.packet_sha256
        == build_datastream_packet(
            waveform=_waveform(),
            spike_raster=_spikes(),
            source_name="unit-replay",
            source_mode="fixture",
            layer_id="sensory",
            qpu_artifact=qpu_artifact,
            optimiser_observation=_observation(),
            metadata={"window": "unit"},
        ).packet_sha256
    )


def test_datastream_packet_rejects_untrusted_shapes_and_values() -> None:
    with pytest.raises(DatastreamValidationError, match="shape"):
        build_datastream_packet(
            waveform=_waveform(),
            spike_raster=np.zeros((7, 4), dtype=np.int8),
            source_name="bad",
            source_mode="fixture",
        )

    bad_spikes = _spikes()
    bad_spikes[0, 0] = 2
    with pytest.raises(DatastreamValidationError, match="binary"):
        build_datastream_packet(
            waveform=_waveform(),
            spike_raster=bad_spikes,
            source_name="bad",
            source_mode="fixture",
        )

    fractional_spikes = _spikes().astype(np.float32)
    fractional_spikes[0, 0] = 0.5
    with pytest.raises(DatastreamValidationError, match="binary"):
        build_datastream_packet(
            waveform=_waveform(),
            spike_raster=fractional_spikes,
            source_name="bad",
            source_mode="fixture",
        )


def test_datastream_payload_validator_rejects_missing_hashes() -> None:
    packet = build_datastream_packet(
        waveform=_waveform(),
        spike_raster=_spikes(),
        source_name="unit-replay",
        source_mode="fixture",
    )
    payload = packet.to_bridge_dict()
    payload["hashes"].pop("aer_bytes_sha256")

    with pytest.raises(DatastreamValidationError, match="aer_bytes_sha256"):
        validate_datastream_payload(payload)
