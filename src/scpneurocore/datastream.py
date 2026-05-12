# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bridge datastream packet for SCPN consumers

"""Build auditable SC-NeuroCore datastream packets for SCPN consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import string
from typing import Any

import numpy as np

from sc_neurocore.edge.telemetry import DeviceTelemetry
from sc_neurocore.optimizer.surrogate_sc_optimizer import BenchmarkObservation
from sc_neurocore.spike_codec.aer_codec import AERCompressionResult, AERSpikeCodec
from sc_neurocore.spike_codec.waveform_codec import WaveformCodec, WaveformCompressionResult
from scpneurocore.bridge import QPUBridgeArtifact, SOURCE_MODES

SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION = "scpneurocore.datastream.v1"


class DatastreamValidationError(ValueError):
    """Raised when a datastream packet cannot be trusted by SCPN consumers."""


@dataclass(frozen=True)
class SCNeuroCoreDatastreamPacket:
    """Hash-addressed bridge packet containing waveform, AER, and telemetry evidence."""

    source_name: str
    source_mode: str
    waveform_shape: tuple[int, int]
    spike_shape: tuple[int, int]
    waveform_codec: str
    waveform_mode: str
    aer_codec: str
    waveform_bytes_sha256: str
    aer_bytes_sha256: str
    waveform_metrics: dict[str, int | float | bool]
    aer_metrics: dict[str, int | float | bool | str]
    telemetry: dict[str, Any]
    qpu_artifact_sha256: str | None = None
    optimiser_observation: dict[str, int | float | str | bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def packet_sha256(self) -> str:
        """Stable hash of the JSON-compatible datastream payload."""
        encoded = json.dumps(
            self.to_bridge_dict(include_packet_hash=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_bridge_dict(self, *, include_packet_hash: bool = True) -> dict[str, Any]:
        """Return the SCPN bridge packet mapping."""
        payload: dict[str, Any] = {
            "schema_version": SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION,
            "source_name": self.source_name,
            "source_mode": self.source_mode,
            "waveform_shape": list(self.waveform_shape),
            "spike_shape": list(self.spike_shape),
            "waveform_codec": self.waveform_codec,
            "waveform_mode": self.waveform_mode,
            "aer_codec": self.aer_codec,
            "hashes": {
                "waveform_bytes_sha256": self.waveform_bytes_sha256,
                "aer_bytes_sha256": self.aer_bytes_sha256,
            },
            "waveform_metrics": dict(self.waveform_metrics),
            "aer_metrics": dict(self.aer_metrics),
            "telemetry": self.telemetry,
            "qpu_artifact_sha256": self.qpu_artifact_sha256,
            "optimiser_observation": (
                None if self.optimiser_observation is None else dict(self.optimiser_observation)
            ),
            "metadata": dict(self.metadata),
        }
        if include_packet_hash:
            payload["packet_sha256"] = self.packet_sha256
        return payload


def build_datastream_packet(
    *,
    waveform: np.ndarray,
    spike_raster: np.ndarray,
    source_name: str,
    source_mode: str,
    layer_id: str = "input",
    waveform_codec: WaveformCodec | None = None,
    aer_codec: AERSpikeCodec | None = None,
    qpu_artifact: QPUBridgeArtifact | None = None,
    optimiser_observation: BenchmarkObservation | None = None,
    metadata: dict[str, Any] | None = None,
) -> SCNeuroCoreDatastreamPacket:
    """Build one audited datastream packet from raw waveform and AER spikes."""
    if source_mode not in SOURCE_MODES:
        raise DatastreamValidationError(f"unsupported source_mode {source_mode!r}")
    waveform_array = _validate_waveform(waveform)
    spikes = _validate_spike_raster(spike_raster)
    if waveform_array.shape != spikes.shape:
        raise DatastreamValidationError(
            f"waveform shape {waveform_array.shape} must match spike_raster shape {spikes.shape}"
        )

    wave_codec = waveform_codec or WaveformCodec(mode="spike")
    event_codec = aer_codec or AERSpikeCodec()
    waveform_bytes, waveform_result = wave_codec.compress(waveform_array)
    aer_bytes, aer_result = event_codec.compress(spikes)
    telemetry = _telemetry_summary(spikes, layer_id=layer_id)
    waveform_shape = (int(waveform_array.shape[0]), int(waveform_array.shape[1]))
    spike_shape = (int(spikes.shape[0]), int(spikes.shape[1]))

    return SCNeuroCoreDatastreamPacket(
        source_name=source_name,
        source_mode=source_mode,
        waveform_shape=waveform_shape,
        spike_shape=spike_shape,
        waveform_codec=type(wave_codec).__name__,
        waveform_mode=wave_codec.mode,
        aer_codec=type(event_codec).__name__,
        waveform_bytes_sha256=_hash_bytes(waveform_bytes),
        aer_bytes_sha256=_hash_bytes(aer_bytes),
        waveform_metrics=_waveform_metrics(waveform_result),
        aer_metrics=_aer_metrics(aer_result),
        telemetry=telemetry,
        qpu_artifact_sha256=None if qpu_artifact is None else qpu_artifact.artifact_sha256,
        optimiser_observation=(
            None if optimiser_observation is None else _observation_record(optimiser_observation)
        ),
        metadata={} if metadata is None else dict(metadata),
    )


def validate_datastream_payload(payload: dict[str, Any]) -> None:
    """Validate the public shape of a bridge datastream payload."""
    if payload.get("schema_version") != SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION:
        raise DatastreamValidationError("unsupported datastream schema_version")
    source_mode = payload.get("source_mode")
    if source_mode not in SOURCE_MODES:
        raise DatastreamValidationError("unsupported source_mode")
    hashes = payload.get("hashes")
    if not isinstance(hashes, dict):
        raise DatastreamValidationError("hashes must be present")
    for key in ("waveform_bytes_sha256", "aer_bytes_sha256"):
        if not isinstance(hashes.get(key), str) or len(hashes[key]) != 64:
            raise DatastreamValidationError(f"{key} must be a SHA256 hex digest")
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, dict) or telemetry.get("total_ticks", 0) <= 0:
        raise DatastreamValidationError("telemetry must contain at least one recorded tick")
    packet_hash = payload.get("packet_sha256")
    if not _is_sha256_hex(packet_hash):
        raise DatastreamValidationError("packet_sha256 must be a SHA256 hex digest")
    expected_hash = _payload_sha256_without_packet_hash(payload)
    if packet_hash != expected_hash:
        raise DatastreamValidationError("packet_sha256 does not match payload")


def _validate_waveform(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim != 2:
        raise DatastreamValidationError(f"waveform must be two-dimensional, got {array.ndim}D")
    if 0 in array.shape:
        raise DatastreamValidationError("waveform dimensions must be non-empty")
    if not np.all(np.isfinite(array)):
        raise DatastreamValidationError("waveform must contain only finite values")
    return array


def _validate_spike_raster(spike_raster: np.ndarray) -> np.ndarray:
    raw = np.asarray(spike_raster)
    if raw.ndim != 2:
        raise DatastreamValidationError(f"spike_raster must be two-dimensional, got {raw.ndim}D")
    if 0 in raw.shape:
        raise DatastreamValidationError("spike_raster dimensions must be non-empty")
    if not np.all((raw == 0) | (raw == 1)):
        raise DatastreamValidationError("spike_raster must contain only binary 0/1 values")
    spikes = raw.astype(np.int8, copy=False)
    if spikes.ndim != 2:
        raise DatastreamValidationError(f"spike_raster must be two-dimensional, got {spikes.ndim}D")
    if 0 in spikes.shape:
        raise DatastreamValidationError("spike_raster dimensions must be non-empty")
    return spikes


def _telemetry_summary(spikes: np.ndarray, *, layer_id: str) -> dict[str, Any]:
    telemetry = DeviceTelemetry()
    n_neurons = int(spikes.shape[1])
    for row in spikes:
        telemetry.record(layer_id, int(np.sum(row)), n_neurons)
    return telemetry.summary()


def _waveform_metrics(result: WaveformCompressionResult) -> dict[str, int | float | bool]:
    return {
        "original_bytes": result.original_bytes,
        "compressed_bytes": result.compressed_bytes,
        "compression_ratio": result.compression_ratio,
        "n_channels": result.n_channels,
        "n_samples": result.n_samples,
        "n_spikes_detected": result.n_spikes_detected,
        "n_templates": result.n_templates,
        "spike_bytes": result.spike_bytes,
        "snippet_bytes": result.snippet_bytes,
        "background_bytes": result.background_bytes,
        "lossless_spikes": result.lossless_spikes,
    }


def _aer_metrics(result: AERCompressionResult) -> dict[str, int | float | bool | str]:
    return {
        "original_bits": result.original_bits,
        "compressed_bits": result.compressed_bits,
        "compression_ratio": result.compression_ratio,
        "n_spikes": result.n_spikes,
        "n_neurons": result.n_neurons,
        "n_timesteps": result.n_timesteps,
        "lossless": result.lossless,
        "n_events": result.n_events,
        "bytes_per_event": result.bytes_per_event,
        "codec_type": result.codec_type,
    }


def _observation_record(observation: BenchmarkObservation) -> dict[str, int | float | str | bool]:
    return {
        "mac_count": observation.mac_count,
        "bitstream_length": observation.bitstream_length,
        "decorrelator": observation.decorrelator,
        "mode": observation.mode,
        "precision_bits": observation.precision_bits,
        "lfsr_polynomial": observation.lfsr_polynomial,
        "luts_used": observation.luts_used,
        "power_mw": observation.power_mw,
        "latency_cycles": observation.latency_cycles,
        "accuracy_score": observation.accuracy_score,
        "is_critical_path": observation.is_critical_path,
    }


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _payload_sha256_without_packet_hash(payload: dict[str, Any]) -> str:
    canonical_payload = dict(payload)
    canonical_payload.pop("packet_sha256", None)
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION",
    "DatastreamValidationError",
    "SCNeuroCoreDatastreamPacket",
    "build_datastream_packet",
    "validate_datastream_payload",
]
