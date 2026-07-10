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
import math
import string
from typing import Any

import numpy as np

from sc_neurocore.edge.telemetry import DeviceTelemetry
from sc_neurocore.optimizer.surrogate_sc_optimizer import BenchmarkObservation
from sc_neurocore.spike_codec.aer_codec import AERCompressionResult, AERSpikeCodec
from sc_neurocore.spike_codec.waveform_codec import WaveformCodec, WaveformCompressionResult
from scpn_neurocore.bridge import QPUBridgeArtifact, SOURCE_MODES

SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION = "scpn_neurocore.datastream.v1"


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
        return _canonical_payload_sha256(self.to_bridge_dict(include_packet_hash=False))

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
    waveform: np.ndarray[Any, Any],
    spike_raster: np.ndarray[Any, Any],
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
    for key in ("source_name", "waveform_codec", "waveform_mode", "aer_codec"):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise DatastreamValidationError(f"{key} must be a non-empty string")
    source_mode = payload.get("source_mode")
    if source_mode not in SOURCE_MODES:
        raise DatastreamValidationError("unsupported source_mode")
    hashes = payload.get("hashes")
    if not isinstance(hashes, dict):
        raise DatastreamValidationError("hashes must be present")
    for key in ("waveform_bytes_sha256", "aer_bytes_sha256"):
        if not _is_sha256_hex(hashes.get(key)):
            raise DatastreamValidationError(f"{key} must be a SHA256 hex digest")

    waveform_shape = _shape_from_payload(payload, "waveform_shape")
    spike_shape = _shape_from_payload(payload, "spike_shape")
    if waveform_shape != spike_shape:
        raise DatastreamValidationError("waveform_shape and spike_shape must match")

    waveform_metrics = _mapping_from_payload(payload, "waveform_metrics")
    aer_metrics = _mapping_from_payload(payload, "aer_metrics")
    telemetry = _mapping_from_payload(payload, "telemetry")
    total_ticks = _positive_int_from_mapping(telemetry, "total_ticks", "telemetry")
    total_spikes = _nonnegative_int_from_mapping(telemetry, "total_spikes", "telemetry")
    _validate_telemetry_layers(
        telemetry,
        expected_total_ticks=total_ticks,
        expected_total_spikes=total_spikes,
    )
    if total_ticks != spike_shape[0]:
        raise DatastreamValidationError("total_ticks must match spike_shape timesteps")
    aer_spikes = _nonnegative_int_from_mapping(aer_metrics, "n_spikes", "aer_metrics")
    if total_spikes != aer_spikes:
        raise DatastreamValidationError("total_spikes must match aer_metrics n_spikes")
    aer_timesteps = _positive_int_from_mapping(aer_metrics, "n_timesteps", "aer_metrics")
    if aer_timesteps != spike_shape[0]:
        raise DatastreamValidationError("n_timesteps must match spike_shape timesteps")
    aer_neurons = _positive_int_from_mapping(aer_metrics, "n_neurons", "aer_metrics")
    if aer_neurons != spike_shape[1]:
        raise DatastreamValidationError("n_neurons must match spike_shape neurons")
    waveform_samples = _positive_int_from_mapping(waveform_metrics, "n_samples", "waveform_metrics")
    if waveform_samples != waveform_shape[0]:
        raise DatastreamValidationError("n_samples must match waveform_shape samples")
    waveform_channels = _positive_int_from_mapping(
        waveform_metrics, "n_channels", "waveform_metrics"
    )
    if waveform_channels != waveform_shape[1]:
        raise DatastreamValidationError("n_channels must match waveform_shape channels")

    qpu_artifact_sha256 = payload.get("qpu_artifact_sha256")
    if qpu_artifact_sha256 is not None and not _is_sha256_hex(qpu_artifact_sha256):
        raise DatastreamValidationError("qpu_artifact_sha256 must be a SHA256 hex digest")
    _validate_optimiser_observation(payload.get("optimiser_observation"))
    if not isinstance(payload.get("metadata"), dict):
        raise DatastreamValidationError("metadata must be a mapping")
    packet_hash = payload.get("packet_sha256")
    if not _is_sha256_hex(packet_hash):
        raise DatastreamValidationError("packet_sha256 must be a SHA256 hex digest")
    expected_hash = _payload_sha256_without_packet_hash(payload)
    if packet_hash != expected_hash:
        raise DatastreamValidationError("packet_sha256 does not match payload")


def _validate_waveform(waveform: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim != 2:
        raise DatastreamValidationError(f"waveform must be two-dimensional, got {array.ndim}D")
    if 0 in array.shape:
        raise DatastreamValidationError("waveform dimensions must be non-empty")
    if not np.all(np.isfinite(array)):
        raise DatastreamValidationError("waveform must contain only finite values")
    return array


def _validate_spike_raster(spike_raster: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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


def _telemetry_summary(spikes: np.ndarray[Any, Any], *, layer_id: str) -> dict[str, Any]:
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


def _shape_from_payload(payload: dict[str, Any], key: str) -> tuple[int, int]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise DatastreamValidationError(f"{key} must be a two-element shape")
    dimensions: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise DatastreamValidationError(f"{key} dimensions must be positive integers")
        dimensions.append(dimension)
    return dimensions[0], dimensions[1]


def _mapping_from_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise DatastreamValidationError(f"{key} must be present")
    return value


def _positive_int_from_mapping(mapping: dict[str, Any], key: str, owner: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatastreamValidationError(f"{key} in {owner} must be a positive integer")
    return value


def _nonnegative_int_from_mapping(mapping: dict[str, Any], key: str, owner: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatastreamValidationError(f"{key} in {owner} must be a non-negative integer")
    return value


def _nonnegative_finite_number_from_mapping(mapping: dict[str, Any], key: str, owner: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise DatastreamValidationError(f"{key} in {owner} must be a finite non-negative number")
    if value < 0.0:
        raise DatastreamValidationError(f"{key} in {owner} must be a finite non-negative number")
    return float(value)


def _validate_telemetry_layers(
    telemetry: dict[str, Any],
    *,
    expected_total_ticks: int,
    expected_total_spikes: int,
) -> None:
    _nonnegative_int_from_mapping(telemetry, "error_count", "telemetry")
    layers = telemetry.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise DatastreamValidationError("layers in telemetry must contain at least one layer")

    layer_tick_total = 0
    layer_spike_total = 0
    for layer_id, record in layers.items():
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise DatastreamValidationError("layer_id in telemetry layers must be non-empty")
        if not isinstance(record, dict):
            raise DatastreamValidationError(f"layers[{layer_id}] must be a telemetry mapping")
        owner = f"layers[{layer_id}]"
        layer_spike_total += _nonnegative_int_from_mapping(record, "spike_count", owner)
        layer_tick_total += _positive_int_from_mapping(record, "tick_count", owner)
        _nonnegative_finite_number_from_mapping(record, "mean_spike_rate", owner)
        mean_utilization = _nonnegative_finite_number_from_mapping(
            record, "mean_utilization", owner
        )
        if mean_utilization > 100.0:
            raise DatastreamValidationError("mean_utilization must not exceed 100 percent")

    if layer_tick_total != expected_total_ticks:
        raise DatastreamValidationError("tick_count layer totals must match telemetry total_ticks")
    if layer_spike_total != expected_total_spikes:
        raise DatastreamValidationError(
            "spike_count layer totals must match telemetry total_spikes"
        )


def _validate_optimiser_observation(observation: Any) -> None:
    if observation is None:
        return
    if not isinstance(observation, dict):
        raise DatastreamValidationError("optimiser_observation must be a mapping or null")

    for key in (
        "mac_count",
        "bitstream_length",
        "precision_bits",
        "luts_used",
        "latency_cycles",
    ):
        _positive_int_from_mapping(observation, key, "optimiser_observation")
    for key in ("decorrelator", "mode", "lfsr_polynomial"):
        if not isinstance(observation.get(key), str) or not observation[key].strip():
            raise DatastreamValidationError(f"{key} in optimiser_observation must be non-empty")
    _nonnegative_finite_number_from_mapping(observation, "power_mw", "optimiser_observation")
    accuracy_score = _nonnegative_finite_number_from_mapping(
        observation, "accuracy_score", "optimiser_observation"
    )
    if accuracy_score > 1.0:
        raise DatastreamValidationError("accuracy_score in optimiser_observation must be in [0, 1]")
    if not isinstance(observation.get("is_critical_path"), bool):
        raise DatastreamValidationError("is_critical_path in optimiser_observation must be boolean")


def _payload_sha256_without_packet_hash(payload: dict[str, Any]) -> str:
    canonical_payload = dict(payload)
    canonical_payload.pop("packet_sha256", None)
    return _canonical_payload_sha256(canonical_payload)


def _canonical_payload_sha256(payload: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DatastreamValidationError("payload must be strict finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION",
    "DatastreamValidationError",
    "SCNeuroCoreDatastreamPacket",
    "build_datastream_packet",
    "validate_datastream_payload",
]
