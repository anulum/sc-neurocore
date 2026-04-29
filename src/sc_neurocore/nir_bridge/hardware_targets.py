# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR hardware capability manifests

"""Capability manifests for NIR-to-neuromorphic-hardware planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SCMappingConstraints:
    """SC-specific constraints used before lowering NIR graphs to a target."""

    bitstream_lengths: tuple[int, ...]
    stream_transport: str
    precision_modes: tuple[str, ...]
    stochastic_sources: tuple[str, ...]
    back_annotation_channels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return {
            "bitstream_lengths": list(self.bitstream_lengths),
            "stream_transport": self.stream_transport,
            "precision_modes": list(self.precision_modes),
            "stochastic_sources": list(self.stochastic_sources),
            "back_annotation_channels": list(self.back_annotation_channels),
        }


@dataclass(frozen=True)
class NeuromorphicHardwareProfile:
    """NIR extension profile for a named neuromorphic target."""

    target_id: str
    display_name: str
    backend_status: str
    supported_nir_nodes: tuple[str, ...]
    unsupported_nir_nodes: tuple[str, ...]
    sc_constraints: SCMappingConstraints
    notes: tuple[str, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        """Return the profile in deterministic manifest form."""

        return {
            "backend_status": self.backend_status,
            "display_name": self.display_name,
            "notes": list(self.notes),
            "sc_constraints": self.sc_constraints.to_dict(),
            "supported_nir_nodes": list(self.supported_nir_nodes),
            "target_id": self.target_id,
            "unsupported_nir_nodes": list(self.unsupported_nir_nodes),
        }


@dataclass(frozen=True)
class HardwareNoiseAnnotation:
    """Measured target noise that can be replayed in simulation."""

    target_id: str
    observations: dict[str, float]
    simulation_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable noise annotation."""

        return {
            "observations": dict(self.observations),
            "simulation_contract": dict(self.simulation_contract),
            "target_id": self.target_id,
        }


_COMMON_SPIKING_NODES = (
    "Input",
    "Output",
    "LIF",
    "IF",
    "LI",
    "CubaLIF",
    "CubaLI",
    "Affine",
    "Linear",
    "Delay",
)

_POOLING_NODES = ("Conv1d", "Conv2d", "SumPool2d", "AvgPool2d", "Flatten")

_PROFILES: dict[str, NeuromorphicHardwareProfile] = {
    "loihi2": NeuromorphicHardwareProfile(
        target_id="loihi2",
        display_name="Loihi 2",
        backend_status="capability_manifest",
        supported_nir_nodes=_COMMON_SPIKING_NODES + _POOLING_NODES,
        unsupported_nir_nodes=("CustomPythonNode",),
        sc_constraints=SCMappingConstraints(
            bitstream_lengths=(64, 128, 256, 512, 1024),
            stream_transport="packet_encoded_probability",
            precision_modes=("unipolar", "bipolar", "mixed_precision"),
            stochastic_sources=("deterministic_lfsr_seed", "sobol_seed"),
            back_annotation_channels=(
                "spike_drop_rate",
                "timing_jitter_ns",
                "threshold_drift",
                "routing_reorder_rate",
            ),
        ),
        notes=("Manifest only; no live SDK invocation is performed.",),
    ),
    "brainscales3": NeuromorphicHardwareProfile(
        target_id="brainscales3",
        display_name="BrainScaleS-3",
        backend_status="capability_manifest",
        supported_nir_nodes=_COMMON_SPIKING_NODES,
        unsupported_nir_nodes=_POOLING_NODES + ("CustomPythonNode",),
        sc_constraints=SCMappingConstraints(
            bitstream_lengths=(128, 256, 512, 1024, 2048),
            stream_transport="analogue_event_probability",
            precision_modes=("unipolar", "mixed_precision"),
            stochastic_sources=("deterministic_lfsr_seed",),
            back_annotation_channels=(
                "membrane_noise",
                "threshold_drift",
                "synapse_mismatch",
                "timing_jitter_ns",
            ),
        ),
        notes=("Analogue mismatch must be measured before replay in simulation.",),
    ),
    "spinnaker2": NeuromorphicHardwareProfile(
        target_id="spinnaker2",
        display_name="SpiNNaker2",
        backend_status="capability_manifest",
        supported_nir_nodes=_COMMON_SPIKING_NODES + _POOLING_NODES,
        unsupported_nir_nodes=("CustomPythonNode",),
        sc_constraints=SCMappingConstraints(
            bitstream_lengths=(32, 64, 128, 256, 512),
            stream_transport="packet_encoded_probability",
            precision_modes=("unipolar", "fixed_point_probability"),
            stochastic_sources=("deterministic_lfsr_seed",),
            back_annotation_channels=(
                "packet_loss_rate",
                "routing_latency_cycles",
                "timing_jitter_ns",
            ),
        ),
        notes=("Packet timing observations can be fed back into event simulation.",),
    ),
    "dynap_se": NeuromorphicHardwareProfile(
        target_id="dynap_se",
        display_name="DYNAP-SE",
        backend_status="capability_manifest",
        supported_nir_nodes=("Input", "Output", "LIF", "IF", "LI", "Delay"),
        unsupported_nir_nodes=("Affine", "Linear") + _POOLING_NODES + ("CustomPythonNode",),
        sc_constraints=SCMappingConstraints(
            bitstream_lengths=(128, 256, 512, 1024),
            stream_transport="aer_probability",
            precision_modes=("unipolar",),
            stochastic_sources=("deterministic_lfsr_seed",),
            back_annotation_channels=(
                "aer_drop_rate",
                "synapse_mismatch",
                "threshold_drift",
            ),
        ),
        notes=("Dense layers require an explicit pre-mapping transform.",),
    ),
}


def available_hardware_profiles() -> tuple[NeuromorphicHardwareProfile, ...]:
    """Return all known hardware profiles in deterministic order."""

    return tuple(_PROFILES[key] for key in sorted(_PROFILES))


def get_hardware_profile(target_id: str) -> NeuromorphicHardwareProfile:
    """Return one hardware profile by identifier."""

    key = target_id.lower().replace("-", "_")
    if key not in _PROFILES:
        known = ", ".join(sorted(_PROFILES))
        raise KeyError(f"unknown neuromorphic target '{target_id}'. Known targets: {known}")
    return _PROFILES[key]


def build_nir_hardware_manifest(targets: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Build a deterministic manifest for NIR hardware-extension planning."""

    selected = tuple(sorted(_PROFILES)) if targets is None else targets
    profiles = [get_hardware_profile(target).to_manifest() for target in selected]
    return {
        "schema_version": "1.0",
        "extension": "sc_neurocore.nir_hardware_targets",
        "profiles": profiles,
    }


def build_noise_annotation(
    target_id: str,
    observations: Mapping[str, float],
) -> HardwareNoiseAnnotation:
    """Validate measured hardware noise and prepare it for simulation replay."""

    profile = get_hardware_profile(target_id)
    allowed = set(profile.sc_constraints.back_annotation_channels)
    unknown = sorted(set(observations) - allowed)
    if unknown:
        raise ValueError(f"unknown noise channels for {profile.target_id}: {', '.join(unknown)}")

    clean: dict[str, float] = {}
    for name, value in observations.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"noise channel '{name}' must be finite and non-negative")
        clean[name] = numeric

    return HardwareNoiseAnnotation(
        target_id=profile.target_id,
        observations=clean,
        simulation_contract={
            "apply_to": "sc_probability_and_event_timing",
            "replay_mode": "deterministic_seeded",
            "requires_measured_hardware": True,
        },
    )
