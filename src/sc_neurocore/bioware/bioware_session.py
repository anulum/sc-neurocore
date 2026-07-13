# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Closed-loop biological-session orchestration

"""Closed-loop biological-session orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from .bioware_acquisition import ArtifactRejector, SpikeDetector, SpikeSorter
from .bioware_analysis import CultureHealth, LatencyBudget
from .bioware_contracts import BioHybridFrameResult, MEAConfig
from .bioware_encoding import (
    AERToSCConverter,
    MEAToAERTranscoder,
    SCToOptoEncoder,
    decode_bitstream_rate,
)
from .bioware_experiment import PharmModel
from .bioware_plasticity import BiologicalSTDP, HomeostaticPlasticity
from .bioware_validation import (
    require_nonnegative,
    require_nonnegative_int,
    validate_voltage_matrix,
)

if TYPE_CHECKING:
    from sc_neurocore.arcane_zenith import ArcaneZenithCognitiveCore


@dataclass
class BioHybridSession:
    """Manages a complete bio-hybrid experiment session.

    Orchestrates MEA recording, spike detection, AER transcoding, stochastic
    conversion, optogenetic feedback, and culture-health assessment. The
    ``stdp`` and ``homeostatic`` policies are retained for caller-managed
    updates; ``process_frame`` does not mutate plasticity state implicitly.
    """

    mea_config: MEAConfig
    detector: SpikeDetector
    transcoder: MEAToAERTranscoder
    sc_converter: AERToSCConverter
    opto_encoder: SCToOptoEncoder
    stdp: BiologicalSTDP = field(default_factory=BiologicalSTDP)
    health_monitor: CultureHealth = field(default_factory=CultureHealth)
    artifact_rejector: Optional["ArtifactRejector"] = None
    pharm_model: Optional["PharmModel"] = None
    latency_budget: Optional["LatencyBudget"] = None
    homeostatic: Optional["HomeostaticPlasticity"] = None
    sorter: Optional["SpikeSorter"] = None
    zenith_core: Optional["ArcaneZenithCognitiveCore"] = None
    round_count: int = 0

    def __post_init__(self) -> None:
        """Validate component compatibility before processing live data."""
        if not isinstance(self.mea_config, MEAConfig):
            raise TypeError("mea_config must be an MEAConfig")
        if not isinstance(self.detector, SpikeDetector):
            raise TypeError("detector must be a SpikeDetector")
        if self.detector.config != self.mea_config:
            raise ValueError("detector.config must match mea_config")
        if not isinstance(self.transcoder, MEAToAERTranscoder):
            raise TypeError("transcoder must be a MEAToAERTranscoder")
        if not isinstance(self.sc_converter, AERToSCConverter):
            raise TypeError("sc_converter must be an AERToSCConverter")
        if not isinstance(self.opto_encoder, SCToOptoEncoder):
            raise TypeError("opto_encoder must be an SCToOptoEncoder")
        if not isinstance(self.stdp, BiologicalSTDP):
            raise TypeError("stdp must be a BiologicalSTDP")
        if not isinstance(self.health_monitor, CultureHealth):
            raise TypeError("health_monitor must be a CultureHealth")
        if self.artifact_rejector is not None and not isinstance(
            self.artifact_rejector, ArtifactRejector
        ):
            raise TypeError("artifact_rejector must be an ArtifactRejector or None")
        if self.pharm_model is not None and not isinstance(self.pharm_model, PharmModel):
            raise TypeError("pharm_model must be a PharmModel or None")
        if self.latency_budget is not None and not isinstance(self.latency_budget, LatencyBudget):
            raise TypeError("latency_budget must be a LatencyBudget or None")
        if self.homeostatic is not None and not isinstance(self.homeostatic, HomeostaticPlasticity):
            raise TypeError("homeostatic must be a HomeostaticPlasticity or None")
        if self.sorter is not None and not isinstance(self.sorter, SpikeSorter):
            raise TypeError("sorter must be a SpikeSorter or None")
        if self.zenith_core is not None and not callable(
            getattr(self.zenith_core, "step_from_bio_rates", None)
        ):
            raise TypeError("zenith_core must provide step_from_bio_rates or be None")
        if self.sc_converter.num_neurons < self.mea_config.num_channels:
            raise ValueError("sc_converter.num_neurons must cover every MEA channel")
        if self.transcoder.channel_map is not None:
            for neuron_id in self.transcoder.channel_map.values():
                if neuron_id >= self.sc_converter.num_neurons:
                    raise ValueError("channel_map targets must fit sc_converter.num_neurons")
        require_nonnegative_int(self.round_count, "round_count")

    def process_frame(
        self,
        voltage_data: np.ndarray[Any, Any],
        t_start_s: float = 0.0,
        stim_times_s: Optional[List[float]] = None,
    ) -> BioHybridFrameResult:
        """Process one frame whose timestamps fit one AER counter epoch.

        Detector timestamps are frame-relative. ``t_start_s`` is the
        non-negative experiment time used by optional experiment models; it
        does not change the frame-local AER timestamp origin.
        """
        validate_voltage_matrix(
            voltage_data,
            expected_channels=self.mea_config.num_channels,
        )
        require_nonnegative(t_start_s, "t_start_s")
        max_frame_tick = int(
            ((voltage_data.shape[0] - 1) / self.mea_config.sample_rate_hz)
            * self.transcoder.hw_clock_hz
        )
        if max_frame_tick > 0xFFFF:
            raise ValueError("voltage frame exceeds the 16-bit AER timestamp epoch")
        if max_frame_tick >= self.sc_converter.window_ticks:
            raise ValueError("voltage frame exceeds sc_converter.window_ticks")

        t0 = time.perf_counter_ns()
        next_round = self.round_count + 1

        if self.artifact_rejector is not None and stim_times_s is not None:
            voltage_data = self.artifact_rejector.blank(
                voltage_data, stim_times_s, self.mea_config.sample_rate_hz
            )

        # 1. Detect spikes
        spikes = self.detector.detect(voltage_data)

        # 1.5 Core primitive wiring
        if self.sorter is not None:
            spikes = self.sorter.assign(spikes)

        if self.pharm_model is not None:
            spikes = self.pharm_model.modulate_spike_events(spikes, t_start_s)

        # 2. Transcode to AER
        aer_events = self.transcoder.transcode(spikes)

        # 3. Convert to SC bitstreams
        bitstreams = self.sc_converter.convert(aer_events)

        # 3.5 Zenith integration!
        if self.zenith_core is not None:
            rates = decode_bitstream_rate(bitstreams)
            self.zenith_core.step_from_bio_rates(rates)

        # 4. Generate optogenetic pulses
        opto_pulses = self.opto_encoder.encode(bitstreams)

        # 5. Health assessment
        n_channels = voltage_data.shape[1]
        spike_counts = np.zeros(n_channels)
        for s in spikes:
            if s.channel < n_channels:
                spike_counts[s.channel] += 1
        duration = voltage_data.shape[0] / self.mea_config.sample_rate_hz
        health = self.health_monitor.assess(spike_counts, duration_s=duration)

        latency_us = (time.perf_counter_ns() - t0) / 1000.0

        result = BioHybridFrameResult(
            round=next_round,
            num_spikes=len(spikes),
            num_aer_events=len(aer_events),
            num_bitstreams=len(bitstreams),
            num_opto_pulses=len(opto_pulses),
            latency_us=latency_us,
            health=health,
            spikes=spikes,
            aer_events=aer_events,
            bitstreams=bitstreams,
            opto_pulses=opto_pulses,
        )
        if self.latency_budget is not None:
            self.latency_budget.record(latency_us)
        self.round_count = next_round
        return result
