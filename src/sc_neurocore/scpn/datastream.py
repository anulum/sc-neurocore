# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN inter-repository datastream contract

"""Deterministic SCPN datastream payloads for cross-repository tests.

The contract is intentionally JSON-friendly. It carries the canonical
``K_nm`` coupling matrix and natural frequencies, a binary spike train
that downstream quantum bridges can consume directly, and derived
rotation angles matching the firing-rate-to-``Ry`` convention used by
the companion quantum-control bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.core.tensor_stream import TensorStream

from .params import N_LAYERS, OMEGA_N, build_knm_matrix

SCHEMA_VERSION = "sc-neurocore.scpn.datastream.v1"
LAYER_IDS = tuple(f"l{i}" for i in range(1, N_LAYERS + 1))


@dataclass(frozen=True)
class SCPNDatastream:
    """In-memory representation of one deterministic SCPN stream."""

    dt_s: float
    seed: int
    probabilities: np.ndarray
    spike_train: np.ndarray
    omega_rad_s: np.ndarray
    knm: np.ndarray

    @property
    def n_steps(self) -> int:
        """Number of timesteps in the stream."""
        return int(self.spike_train.shape[0])

    @property
    def n_layers(self) -> int:
        """Number of SCPN layer channels in the stream."""
        return int(self.spike_train.shape[1])

    @property
    def firing_rates(self) -> np.ndarray:
        """Mean spike probability per layer over this stream window."""
        return self.spike_train.astype(np.float64).mean(axis=0)

    @property
    def rotation_angles_rad(self) -> np.ndarray:
        """``Ry`` angles for quantum-control bridges: firing rate times pi."""
        return self.firing_rates * np.pi

    @property
    def quantum_amplitudes(self) -> np.ndarray:
        """Real amplitude encoding of firing rates as ``[alpha, beta]`` pairs."""
        amplitudes = TensorStream.from_prob(self.firing_rates).to_quantum()
        return np.real(amplitudes).astype(np.float64)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialise the datastream to a stable JSON-compatible mapping."""
        validate_scpn_datastream(self)
        return {
            "schema_version": SCHEMA_VERSION,
            "source_project": "sc-neurocore",
            "dt_s": self.dt_s,
            "seed": self.seed,
            "n_steps": self.n_steps,
            "n_layers": self.n_layers,
            "layer_ids": list(LAYER_IDS),
            "omega_rad_s": self.omega_rad_s.tolist(),
            "knm": self.knm.tolist(),
            "probabilities": self.probabilities.tolist(),
            "spike_train": self.spike_train.astype(int).tolist(),
            "firing_rates": self.firing_rates.tolist(),
            "rotation_angles_rad": self.rotation_angles_rad.tolist(),
            "quantum_amplitudes": self.quantum_amplitudes.tolist(),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> SCPNDatastream:
        """Load and validate a datastream from a JSON-compatible mapping."""
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported SCPN datastream schema version")
        stream = cls(
            dt_s=float(payload["dt_s"]),
            seed=int(payload["seed"]),
            probabilities=np.asarray(payload["probabilities"], dtype=np.float64),
            spike_train=np.asarray(payload["spike_train"], dtype=np.uint8),
            omega_rad_s=np.asarray(payload["omega_rad_s"], dtype=np.float64),
            knm=np.asarray(payload["knm"], dtype=np.float64),
        )
        validate_scpn_datastream(stream)
        return stream


def generate_scpn_datastream(
    *,
    n_steps: int = 32,
    dt_s: float = 0.01,
    seed: int = 1729,
    spike_floor: float = 0.02,
    spike_ceiling: float = 0.98,
) -> SCPNDatastream:
    """Generate a deterministic 16-layer stream for inter-repository tests.

    The probability envelope is a bounded phase oscillator driven by the
    canonical SCPN natural frequencies and a small normalised coupling
    bias derived from ``K_nm``. The binary spike train is sampled from
    that envelope with a local RNG seeded by ``seed``.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0")
    if not 0.0 <= spike_floor < spike_ceiling <= 1.0:
        raise ValueError("spike_floor and spike_ceiling must satisfy 0 <= floor < ceiling <= 1")

    omega = np.asarray(OMEGA_N, dtype=np.float64)
    knm = build_knm_matrix()
    coupling = knm.sum(axis=1)
    coupling_span = float(np.ptp(coupling))
    if coupling_span > 0.0:
        coupling_bias = (coupling - coupling.min()) / coupling_span
    else:
        coupling_bias = np.zeros_like(coupling)
    coupling_bias = coupling_bias - float(coupling_bias.mean())

    t = np.arange(n_steps, dtype=np.float64)[:, None] * dt_s
    phase = t * omega[None, :]
    probabilities = 0.5 + 0.25 * np.sin(phase) + 0.08 * coupling_bias[None, :]
    probabilities = np.clip(probabilities, spike_floor, spike_ceiling)

    rng = np.random.default_rng(seed)
    spike_train = (rng.random(probabilities.shape) < probabilities).astype(np.uint8)

    stream = SCPNDatastream(
        dt_s=dt_s,
        seed=seed,
        probabilities=probabilities,
        spike_train=spike_train,
        omega_rad_s=omega.copy(),
        knm=knm,
    )
    validate_scpn_datastream(stream)
    return stream


def validate_scpn_datastream(stream: SCPNDatastream) -> None:
    """Validate shape, bounds, and canonical matrix invariants."""
    if stream.dt_s <= 0.0:
        raise ValueError("dt_s must be > 0")
    if stream.probabilities.shape != stream.spike_train.shape:
        raise ValueError("probabilities and spike_train must have matching shapes")
    if stream.probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2-D array")
    if stream.spike_train.shape[1] != N_LAYERS:
        raise ValueError(f"spike_train must have {N_LAYERS} layer columns")
    if stream.omega_rad_s.shape != (N_LAYERS,):
        raise ValueError(f"omega_rad_s must have shape ({N_LAYERS},)")
    if stream.knm.shape != (N_LAYERS, N_LAYERS):
        raise ValueError(f"knm must have shape ({N_LAYERS}, {N_LAYERS})")
    if not np.all((stream.probabilities >= 0.0) & (stream.probabilities <= 1.0)):
        raise ValueError("probabilities must be in [0, 1]")
    if not np.isin(stream.spike_train, [0, 1]).all():
        raise ValueError("spike_train must be binary")
    if not np.allclose(stream.knm, stream.knm.T, atol=1e-12):
        raise ValueError("knm must be symmetric")
    if not np.allclose(np.diag(stream.knm), 0.0, atol=1e-12):
        raise ValueError("knm diagonal must be zero")


def write_scpn_datastream(path: str | Path, stream: SCPNDatastream) -> None:
    """Write a stream payload to JSON."""
    path = Path(path)
    path.write_text(json.dumps(stream.to_json_dict(), indent=2, sort_keys=True) + "\n")


def read_scpn_datastream(path: str | Path) -> SCPNDatastream:
    """Read a stream payload from JSON."""
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("SCPN datastream JSON root must be an object")
    return SCPNDatastream.from_json_dict(payload)


def generate_scpn_datastream_payload(**kwargs: Any) -> dict[str, Any]:
    """Generate a JSON-compatible stream payload in one call."""
    return generate_scpn_datastream(**kwargs).to_json_dict()


__all__ = [
    "SCHEMA_VERSION",
    "SCPNDatastream",
    "generate_scpn_datastream",
    "generate_scpn_datastream_payload",
    "read_scpn_datastream",
    "validate_scpn_datastream",
    "write_scpn_datastream",
]
