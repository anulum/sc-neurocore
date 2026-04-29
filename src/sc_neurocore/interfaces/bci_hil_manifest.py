# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BCI HIL reference manifests

"""Reference manifests for closed-loop BCI hardware-in-the-loop templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sc_neurocore.interfaces.bci_closed_loop import ClosedLoopBCIConfig, ClosedLoopBCITemplate


@dataclass(frozen=True)
class BCIHILBoardProfile:
    """Board/input profile for a closed-loop BCI reference pipeline."""

    profile_id: str
    display_name: str
    board: str
    input_source: str
    model_reference: str
    n_channels: int
    sampling_rate_hz: int
    transport: str
    feedback_transport: str
    required_artefacts: tuple[str, ...]
    pipeline_steps: tuple[str, ...]
    safety_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic manifest dictionary."""

        return {
            "board": self.board,
            "display_name": self.display_name,
            "feedback_transport": self.feedback_transport,
            "input_source": self.input_source,
            "model_reference": self.model_reference,
            "n_channels": self.n_channels,
            "pipeline_steps": list(self.pipeline_steps),
            "profile_id": self.profile_id,
            "required_artefacts": list(self.required_artefacts),
            "safety_contract": dict(self.safety_contract),
            "sampling_rate_hz": self.sampling_rate_hz,
            "transport": self.transport,
        }


_PIPELINE_STEPS = (
    "raw_waveform_window",
    "waveform_codec_compression",
    "threshold_spike_raster",
    "aer_payload_generation",
    "rate_decoder",
    "feedback_frame",
    "device_telemetry",
)

_REFERENCE_PROFILES: dict[str, BCIHILBoardProfile] = {
    "pynq_shd": BCIHILBoardProfile(
        profile_id="pynq_shd",
        display_name="PYNQ SHD closed-loop reference",
        board="PYNQ-Z2 / Zynq XC7Z020",
        input_source="SHD 700-channel event-raster emulator",
        model_reference="sc_neurocore.model_zoo.configs.shd_speech_classifier",
        n_channels=700,
        sampling_rate_hz=30_000,
        transport="PYNQ AXI control plane plus AER event payloads",
        feedback_transport="implant emulator or PYNQ register adapter",
        required_artefacts=(
            "model_zoo shd_speech_classifier topology",
            "ClosedLoopBCITemplate runtime",
            "WaveformCodec spike mode",
            "AERSpikeCodec payload",
            "DeviceTelemetry summary",
        ),
        pipeline_steps=_PIPELINE_STEPS,
        safety_contract={
            "hardware_required": False,
            "default_feedback_sink": "implant_emulator",
            "requires_external_bitstream_for_physical_board": True,
            "no_stimulation_without_sink_override": True,
        },
    ),
    "probe_384ch": BCIHILBoardProfile(
        profile_id="probe_384ch",
        display_name="384-channel probe closed-loop emulator",
        board="host emulator",
        input_source="384-channel extracellular probe window",
        model_reference="rate_decoder_reference",
        n_channels=384,
        sampling_rate_hz=30_000,
        transport="in-process waveform windows",
        feedback_transport="implant emulator",
        required_artefacts=(
            "ClosedLoopBCITemplate runtime",
            "WaveformCodec spike mode",
            "AERSpikeCodec payload",
            "DeviceTelemetry summary",
        ),
        pipeline_steps=_PIPELINE_STEPS,
        safety_contract={
            "hardware_required": False,
            "default_feedback_sink": "implant_emulator",
            "requires_external_bitstream_for_physical_board": False,
            "no_stimulation_without_sink_override": True,
        },
    ),
}


def available_bci_hil_profiles() -> tuple[BCIHILBoardProfile, ...]:
    """Return all reference profiles in deterministic order."""

    return tuple(_REFERENCE_PROFILES[key] for key in sorted(_REFERENCE_PROFILES))


def get_bci_hil_profile(profile_id: str) -> BCIHILBoardProfile:
    """Return one reference profile by identifier."""

    key = profile_id.lower().replace("-", "_")
    if key not in _REFERENCE_PROFILES:
        known = ", ".join(sorted(_REFERENCE_PROFILES))
        raise KeyError(f"unknown BCI HIL profile '{profile_id}'. Known profiles: {known}")
    return _REFERENCE_PROFILES[key]


def build_bci_hil_reference_manifest(profile_id: str = "pynq_shd") -> dict[str, Any]:
    """Build a deterministic closed-loop BCI/HIL reference manifest."""

    profile = get_bci_hil_profile(profile_id)
    config = _profile_to_config(profile)
    return {
        "schema_version": "1.0",
        "profile": profile.to_dict(),
        "template_config": {
            "feedback_gain": config.feedback_gain,
            "feedback_layer_id": config.feedback_layer_id,
            "input_layer_id": config.input_layer_id,
            "max_feedback": config.max_feedback,
            "n_channels": config.n_channels,
            "quantize_bits": config.quantize_bits,
            "sampling_rate_hz": config.sampling_rate_hz,
            "snippet_samples": config.snippet_samples,
            "threshold_sigma": config.threshold_sigma,
            "timestamp_bits": config.timestamp_bits,
            "waveform_mode": config.waveform_mode,
        },
    }


def create_bci_hil_template(profile_id: str = "pynq_shd") -> ClosedLoopBCITemplate:
    """Create a `ClosedLoopBCITemplate` from a reference profile."""

    return ClosedLoopBCITemplate(_profile_to_config(get_bci_hil_profile(profile_id)))


def _profile_to_config(profile: BCIHILBoardProfile) -> ClosedLoopBCIConfig:
    return ClosedLoopBCIConfig(
        n_channels=profile.n_channels,
        sampling_rate_hz=profile.sampling_rate_hz,
        threshold_sigma=4.0,
        snippet_samples=16,
        waveform_mode="spike",
        quantize_bits=6,
        timestamp_bits=16,
        feedback_gain=1.0,
        max_feedback=1.0,
        input_layer_id=f"{profile.profile_id}_input",
        feedback_layer_id=f"{profile.profile_id}_feedback",
    )
