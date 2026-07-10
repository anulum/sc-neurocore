# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-control campaign bridge compatibility

"""Auditable QPU data artifacts for quantum-control campaign bridges."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import string
from typing import Any

import numpy as np

from sc_neurocore.scpn.params import build_knm_matrix

QPU_ARTIFACT_SCHEMA_VERSION = "scpn-quantum-control.qpu-data-artifact.v1"
PUBLICATION_SOURCE_MODES = frozenset({"recorded", "replay", "curated", "derived"})
NON_PUBLICATION_SOURCE_MODES = frozenset({"synthetic", "simulation", "fixture"})
SOURCE_MODES = PUBLICATION_SOURCE_MODES | NON_PUBLICATION_SOURCE_MODES


class SourceDataUnavailable(FileNotFoundError):
    """Raised when a requested publication-grade source is not bundled."""


@dataclass(frozen=True)
class QPUBridgeArtifact:
    """Provenance-rich oscillator artifact for QPU campaign consumers."""

    domain: str
    source_name: str
    source_mode: str
    K_nm: np.ndarray[Any, Any]
    omega: np.ndarray[Any, Any]
    theta0: np.ndarray[Any, Any] | None
    layer_assignments: list[int]
    normalization: str
    extraction_method: str
    source_timestamp: str | None = None
    replay_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_source_mode(self.source_mode)
        for key, value in (
            ("domain", self.domain),
            ("source_name", self.source_name),
            ("normalization", self.normalization),
            ("extraction_method", self.extraction_method),
        ):
            _require_non_empty_string(value, key)
        _validate_artifact_arrays(self.K_nm, self.omega, self.theta0, self.layer_assignments)
        if self.source_timestamp is None and self.replay_id is None:
            raise ValueError("source_timestamp or replay_id is required")
        if self.source_timestamp is not None:
            _require_non_empty_string(self.source_timestamp, "source_timestamp")
        if self.replay_id is not None:
            _require_non_empty_string(self.replay_id, "replay_id")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")

    @property
    def hashes(self) -> dict[str, str]:
        """Stable SHA256 hashes for numeric payloads."""
        result = {
            "K_nm_sha256": _hash_array(self.K_nm),
            "omega_sha256": _hash_array(self.omega),
        }
        if self.theta0 is not None:
            result["theta0_sha256"] = _hash_array(self.theta0)
        return result

    @property
    def artifact_sha256(self) -> str:
        """Stable SHA256 of the full JSON-compatible artifact payload."""
        payload = self.to_qpu_artifact_dict(include_artifact_hash=False)
        return _canonical_artifact_sha256(payload)

    def to_qpu_artifact_dict(self, *, include_artifact_hash: bool = True) -> dict[str, Any]:
        """Return the Quantum Control artifact mapping."""
        payload: dict[str, Any] = {
            "schema_version": QPU_ARTIFACT_SCHEMA_VERSION,
            "domain": self.domain,
            "source_name": self.source_name,
            "source_mode": self.source_mode,
            "K_nm": self.K_nm.tolist(),
            "omega": self.omega.tolist(),
            "theta0": None if self.theta0 is None else self.theta0.tolist(),
            "layer_assignments": list(self.layer_assignments),
            "normalization": self.normalization,
            "extraction_method": self.extraction_method,
            "source_timestamp": self.source_timestamp,
            "replay_id": self.replay_id,
            "metadata": dict(self.metadata),
            "hashes": self.hashes,
        }
        if include_artifact_hash:
            payload["artifact_sha256"] = self.artifact_sha256
        return payload


def load_connectome(
    name: str,
    n: int | None = None,
    *,
    source_mode: str | None = None,
    synthetic: bool = False,
) -> QPUBridgeArtifact:
    """Load a connectome artifact for Quantum Control.

    Publication-grade connectome files are not bundled in this repository.
    Callers must provide a real source through a future curated/replay path,
    or explicitly request a labelled synthetic smoke artifact.
    """
    if name not in {"c_elegans_sub", "c_elegans"}:
        raise ValueError(f"unsupported connectome source {name!r}")
    mode = _resolve_source_mode(source_mode, synthetic)
    size = 14 if n is None else _require_positive_n(n)
    if mode not in NON_PUBLICATION_SOURCE_MODES:
        _raise_unavailable(name, (size, size))

    knm = _chain_coupling(size, nearest=0.62, next_nearest=0.22)
    omega = np.linspace(0.8, 1.8, size, dtype=np.float64)
    theta0 = np.linspace(0.0, np.pi, size, endpoint=False, dtype=np.float64)
    return _artifact(
        domain="connectome",
        source_name=name,
        source_mode=mode,
        knm=knm,
        omega=omega,
        theta0=theta0,
        normalization="max_abs_to_unit_interval",
        extraction_method="deterministic_chain_smoke_fixture",
        metadata={
            "expected_shape": [size, size],
            "publication_safe": False,
            "reason": "synthetic smoke artifact; no bundled connectome source used",
        },
    )


def load_tokamak_data(
    n: int = 16,
    *,
    source_mode: str | None = None,
    synthetic: bool = False,
) -> QPUBridgeArtifact:
    """Load tokamak/plasma oscillator data for Quantum Control."""
    mode = _resolve_source_mode(source_mode, synthetic)
    size = _require_positive_n(n)
    if mode not in NON_PUBLICATION_SOURCE_MODES:
        _raise_unavailable("tokamak", (size, size))

    knm: np.ndarray[Any, Any] = _banded_coupling(size, base=0.45, decay=0.32)
    omega: np.ndarray[Any, Any] = np.resize(
        np.array([10.0, 8.0, 3.0, 5.0, 0.5, 0.3, 0.1, 1.0], dtype=np.float64),
        size,
    )
    return _artifact(
        domain="tokamak",
        source_name="tokamak",
        source_mode=mode,
        knm=knm,
        omega=omega,
        theta0=np.zeros(size, dtype=np.float64),
        normalization="bounded_exponential_coupling",
        extraction_method="deterministic_plasma_timescale_smoke_fixture",
        metadata={
            "expected_shape": [size, size],
            "omega_units": "rad_s",
            "publication_safe": False,
        },
    )


def load_power_grid(
    n: int,
    name: str | None = None,
    *,
    source_mode: str | None = None,
    synthetic: bool = False,
) -> QPUBridgeArtifact:
    """Load power-grid oscillator data for Quantum Control."""
    mode = _resolve_source_mode(source_mode, synthetic)
    size = _require_positive_n(n)
    source_name = "power_grid" if name is None else name
    if mode not in NON_PUBLICATION_SOURCE_MODES:
        _raise_unavailable(source_name, (size, size))

    knm: np.ndarray[Any, Any] = _ring_coupling(size, nearest=0.5, long_range=0.08)
    omega: np.ndarray[Any, Any] = np.ones(size, dtype=np.float64)
    if size >= 4:
        omega[1::4] = 1.02
        omega[3::4] = 0.98
    return _artifact(
        domain="power_grid",
        source_name=source_name,
        source_mode=mode,
        knm=knm,
        omega=omega,
        theta0=np.zeros(size, dtype=np.float64),
        normalization="per_unit_admittance_like_smoke_scaling",
        extraction_method="deterministic_ring_grid_smoke_fixture",
        metadata={
            "expected_shape": [size, size],
            "omega_units": "per_unit_frequency",
            "publication_safe": False,
        },
    )


def load_live_stream(
    source: str,
    step: int,
    *,
    source_mode: str | None = None,
    synthetic: bool = False,
) -> QPUBridgeArtifact:
    """Load one replayable live-stream artifact."""
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if source != "eeg_powergrid":
        raise ValueError(f"unsupported live stream source {source!r}")
    mode = _resolve_source_mode(source_mode, synthetic)
    size = 12
    if mode not in NON_PUBLICATION_SOURCE_MODES:
        _raise_unavailable(source, (size, size))

    knm = _ring_coupling(size, nearest=0.38, long_range=0.04)
    phase = float(step) * 0.1
    omega = 1.0 + 0.05 * np.sin(phase + np.arange(size, dtype=np.float64) * 0.5)
    theta0 = (phase + np.arange(size, dtype=np.float64) * np.pi / size) % (2.0 * np.pi)
    return _artifact(
        domain="live_stream",
        source_name=source,
        source_mode=mode,
        knm=knm,
        omega=omega,
        theta0=theta0,
        normalization="bounded_live_replay_smoke_scaling",
        extraction_method="deterministic_eeg_powergrid_step_fixture",
        replay_id=f"{mode}:{source}:step:{step}",
        metadata={
            "step": step,
            "expected_shape": [size, size],
            "publication_safe": False,
        },
    )


def validate_qpu_artifact_payload(payload: dict[str, Any]) -> None:
    """Validate a JSON-deserialised QPU bridge artifact payload."""
    if not isinstance(payload, dict):
        raise ValueError("QPU artifact payload must be a mapping")
    if payload.get("schema_version") != QPU_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported QPU artifact schema_version")
    for key in ("domain", "source_name", "normalization", "extraction_method"):
        _require_non_empty_string(payload.get(key), key)
    source_mode = payload.get("source_mode")
    _validate_source_mode(source_mode)
    source_timestamp = payload.get("source_timestamp")
    replay_id = payload.get("replay_id")
    if source_timestamp is None and replay_id is None:
        raise ValueError("source_timestamp or replay_id is required")
    if source_timestamp is not None:
        _require_non_empty_string(source_timestamp, "source_timestamp")
    if replay_id is not None:
        _require_non_empty_string(replay_id, "replay_id")
    if not isinstance(payload.get("metadata"), dict):
        raise ValueError("metadata must be a mapping")

    knm = _array_from_payload(payload, "K_nm")
    omega = _array_from_payload(payload, "omega")
    theta0 = None if payload.get("theta0") is None else _array_from_payload(payload, "theta0")
    layer_assignments = payload.get("layer_assignments")
    if not isinstance(layer_assignments, list):
        raise ValueError("layer_assignments must be a list")
    _validate_artifact_arrays(knm, omega, theta0, layer_assignments)

    hashes = payload.get("hashes")
    if not isinstance(hashes, dict):
        raise ValueError("hashes must be present")
    _validate_payload_array_hash(hashes, "K_nm_sha256", knm)
    _validate_payload_array_hash(hashes, "omega_sha256", omega)
    if theta0 is None:
        if "theta0_sha256" in hashes:
            raise ValueError("theta0_sha256 must be absent when theta0 is null")
    else:
        _validate_payload_array_hash(hashes, "theta0_sha256", theta0)

    artifact_hash = payload.get("artifact_sha256")
    if not _is_sha256_hex(artifact_hash):
        raise ValueError("artifact_sha256 must be a SHA256 hex digest")
    expected_hash = _payload_sha256_without_artifact_hash(payload)
    if artifact_hash != expected_hash:
        raise ValueError("artifact_sha256 does not match payload")


def _artifact(
    *,
    domain: str,
    source_name: str,
    source_mode: str,
    knm: np.ndarray[Any, Any],
    omega: np.ndarray[Any, Any],
    theta0: np.ndarray[Any, Any] | None,
    normalization: str,
    extraction_method: str,
    metadata: dict[str, Any],
    replay_id: str | None = None,
) -> QPUBridgeArtifact:
    size = int(knm.shape[0])
    return QPUBridgeArtifact(
        domain=domain,
        source_name=source_name,
        source_mode=source_mode,
        K_nm=knm,
        omega=omega,
        theta0=theta0,
        layer_assignments=list(range(size)),
        normalization=normalization,
        extraction_method=extraction_method,
        replay_id=replay_id or f"{source_mode}:{domain}:{source_name}:n{size}",
        metadata=metadata,
    )


def _resolve_source_mode(source_mode: str | None, synthetic: bool) -> str:
    if synthetic:
        return "synthetic"
    if source_mode is None:
        raise SourceDataUnavailable(
            "source_mode is required; use a publication source mode with available data "
            "or explicitly request source_mode='synthetic' for smoke tests"
        )
    _validate_source_mode(source_mode)
    return source_mode


def _validate_source_mode(source_mode: object) -> None:
    if source_mode not in SOURCE_MODES:
        raise ValueError(f"unsupported source_mode {source_mode!r}")


def _require_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _raise_unavailable(source_name: str, expected_shape: tuple[int, int]) -> None:
    raise SourceDataUnavailable(
        f"source {source_name!r} is unavailable; expected K_nm shape {expected_shape}"
    )


def _require_positive_n(n: int) -> int:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return n


def _validate_artifact_arrays(
    knm: np.ndarray[Any, Any],
    omega: np.ndarray[Any, Any],
    theta0: np.ndarray[Any, Any] | None,
    layer_assignments: list[int],
) -> None:
    if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
        raise ValueError(f"K_nm must be square, got shape {knm.shape}")
    n = int(knm.shape[0])
    if omega.shape != (n,):
        raise ValueError(f"omega must have shape ({n},), got {omega.shape}")
    if theta0 is not None and theta0.shape != (n,):
        raise ValueError(f"theta0 must have shape ({n},), got {theta0.shape}")
    if len(layer_assignments) != n:
        raise ValueError(f"layer_assignments must have length {n}")
    if any(
        isinstance(layer_id, bool) or not isinstance(layer_id, int) or layer_id < 0
        for layer_id in layer_assignments
    ):
        raise ValueError("layer_assignments must be non-negative integer ids")
    if len(set(layer_assignments)) != n:
        raise ValueError("layer_assignments must be unique")
    if not np.all(np.isfinite(knm)):
        raise ValueError("K_nm must contain only finite values")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must contain only finite values")
    if theta0 is not None and not np.all(np.isfinite(theta0)):
        raise ValueError("theta0 must contain only finite values")
    if not np.allclose(knm, knm.T, atol=1e-12):
        raise ValueError("K_nm must be symmetric")
    if not np.allclose(np.diag(knm), 0.0, atol=1e-12):
        raise ValueError("K_nm diagonal must be zero")
    if np.any(knm < 0.0):
        raise ValueError("K_nm must be non-negative")


def _hash_array(array: np.ndarray[Any, Any]) -> str:
    stable = np.ascontiguousarray(array, dtype=np.float64)
    return hashlib.sha256(stable.tobytes()).hexdigest()


def _array_from_payload(payload: dict[str, Any], key: str) -> np.ndarray[Any, Any]:
    try:
        return np.asarray(payload.get(key), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a numeric JSON array") from exc


def _validate_payload_array_hash(
    hashes: dict[str, Any], key: str, array: np.ndarray[Any, Any]
) -> None:
    value = hashes.get(key)
    if not _is_sha256_hex(value):
        raise ValueError(f"{key} must be a SHA256 hex digest")
    if value != _hash_array(array):
        raise ValueError(f"{key} does not match payload")


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in string.hexdigits for character in value)
    )


def _payload_sha256_without_artifact_hash(payload: dict[str, Any]) -> str:
    canonical_payload = dict(payload)
    canonical_payload.pop("artifact_sha256", None)
    return _canonical_artifact_sha256(canonical_payload)


def _canonical_artifact_sha256(payload: dict[str, Any]) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact payload must be strict finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _banded_coupling(n: int, *, base: float, decay: float) -> np.ndarray[Any, Any]:
    if n == 16:
        return build_knm_matrix()
    idx: np.ndarray[Any, Any] = np.arange(n, dtype=np.float64)
    dist: np.ndarray[Any, Any] = np.abs(idx[:, None] - idx[None, :])
    knm: np.ndarray[Any, Any] = base * np.exp(-decay * dist)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


def _chain_coupling(n: int, *, nearest: float, next_nearest: float) -> np.ndarray[Any, Any]:
    knm: np.ndarray[Any, Any] = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        knm[i, i + 1] = nearest
        knm[i + 1, i] = nearest
    for i in range(n - 2):
        knm[i, i + 2] = next_nearest
        knm[i + 2, i] = next_nearest
    if n > 1:
        knm /= float(np.max(knm))
    return knm


def _ring_coupling(n: int, *, nearest: float, long_range: float) -> np.ndarray[Any, Any]:
    knm: np.ndarray[Any, Any] = np.zeros((n, n), dtype=np.float64)
    if n == 1:
        return knm
    for i in range(n):
        j = (i + 1) % n
        knm[i, j] = nearest
        knm[j, i] = nearest
    if n > 4:
        half = n // 2
        for i in range(n):
            j = (i + half) % n
            knm[i, j] = max(knm[i, j], long_range)
            knm[j, i] = max(knm[j, i], long_range)
    return knm


__all__ = [
    "QPU_ARTIFACT_SCHEMA_VERSION",
    "QPUBridgeArtifact",
    "SourceDataUnavailable",
    "load_connectome",
    "load_live_stream",
    "load_power_grid",
    "load_tokamak_data",
    "validate_qpu_artifact_payload",
]
