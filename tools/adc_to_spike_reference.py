# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - bit-true ADC-to-spike quantiser reference.

"""Bit-true reference model for the ADC-to-spike HDL quantiser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ADCSpikeConfig:
    adc_width: int = 16
    q_int: int = 8
    q_frac: int = 8
    decimation: int = 8
    signed_input: bool = True
    threshold_q: int = 256
    base_address: int = 0
    negative_offset: int = 1

    @property
    def q_total(self) -> int:
        return self.q_int + self.q_frac

    @property
    def q_min(self) -> int:
        return -(1 << (self.q_total - 1))

    @property
    def q_max(self) -> int:
        return (1 << (self.q_total - 1)) - 1

    def validate(self) -> None:
        if self.adc_width <= 1:
            raise ValueError("adc_width must be greater than one")
        if self.q_int <= 0 or self.q_frac < 0:
            raise ValueError(
                "Q-format must have positive integer bits and non-negative fraction bits"
            )
        if self.decimation <= 0:
            raise ValueError("decimation must be positive")
        if self.threshold_q <= 0:
            raise ValueError("threshold_q must be positive")


@dataclass(frozen=True, slots=True)
class ADCSpikeStep:
    accepted_sample: bool
    adc_ready: bool
    aer_valid: bool
    aer_address: int | None
    aer_polarity: int | None
    window_q: int
    pending_spikes: int
    dropped_sample: bool
    threshold_error: bool
    sample_count: int
    spike_count: int


class ADCToSpikeReference:
    """Cycle-stepped ADC decimator and deterministic rate-code generator."""

    def __init__(self, config: ADCSpikeConfig) -> None:
        config.validate()
        self.config = config
        self.decim_count = 0
        self.window_sum_q = 0
        self.pending_spikes = 0
        self.pending_polarity = 0
        self.last_window_q = 0
        self.dropped_sample = False
        self.threshold_error = False
        self.sample_count = 0
        self.spike_count = 0

    @property
    def adc_ready(self) -> bool:
        return self.pending_spikes == 0 and self.config.threshold_q > 0

    @property
    def aer_valid(self) -> bool:
        return self.pending_spikes > 0

    @property
    def aer_address(self) -> int:
        if self.pending_polarity:
            return self.config.base_address + self.config.negative_offset
        return self.config.base_address

    def quantise_adc(self, sample: int) -> int:
        cfg = self.config
        if cfg.signed_input:
            sign_bit = 1 << (cfg.adc_width - 1)
            mask = (1 << cfg.adc_width) - 1
            sample &= mask
            centred = sample - (1 << cfg.adc_width) if sample & sign_bit else sample
        else:
            centred = sample - (1 << (cfg.adc_width - 1))

        if cfg.q_total > cfg.adc_width:
            rounded = centred << (cfg.q_total - cfg.adc_width)
        elif cfg.adc_width > cfg.q_total:
            shift = cfg.adc_width - cfg.q_total
            half = 1 << (shift - 1)
            rounded = (centred + half) >> shift if centred >= 0 else (centred - half) >> shift
        else:
            rounded = centred
        return max(cfg.q_min, min(cfg.q_max, rounded))

    def _average_window(self, total_q: int) -> int:
        half = self.config.decimation // 2
        adjusted = total_q + half if total_q >= 0 else total_q - half
        averaged = int(adjusted / self.config.decimation)
        return max(self.config.q_min, min(self.config.q_max, averaged))

    def step(self, sample: int | None, *, adc_valid: bool, aer_ready: bool) -> ADCSpikeStep:
        accepted_spike = self.aer_valid and aer_ready
        emitted_address = self.aer_address if self.aer_valid else None
        emitted_polarity = self.pending_polarity if self.aer_valid else None

        if accepted_spike:
            self.pending_spikes -= 1
            self.spike_count += 1

        ready = self.adc_ready
        accepted_sample = bool(adc_valid and ready)
        if adc_valid and not ready:
            self.dropped_sample = True

        if accepted_sample:
            assert sample is not None
            q_sample = self.quantise_adc(sample)
            next_sum = self.window_sum_q + q_sample
            self.sample_count += 1
            if self.decim_count == self.config.decimation - 1:
                self.last_window_q = self._average_window(next_sum)
                self.pending_spikes = abs(self.last_window_q) // self.config.threshold_q
                self.pending_polarity = 1 if self.last_window_q < 0 else 0
                self.window_sum_q = 0
                self.decim_count = 0
            else:
                self.window_sum_q = next_sum
                self.decim_count += 1

        return ADCSpikeStep(
            accepted_sample=accepted_sample,
            adc_ready=ready,
            aer_valid=emitted_address is not None,
            aer_address=emitted_address,
            aer_polarity=emitted_polarity,
            window_q=self.last_window_q,
            pending_spikes=self.pending_spikes,
            dropped_sample=self.dropped_sample,
            threshold_error=self.threshold_error,
            sample_count=self.sample_count,
            spike_count=self.spike_count,
        )

    def run(self, samples: Iterable[int], *, drain: bool = True) -> list[ADCSpikeStep]:
        steps: list[ADCSpikeStep] = []
        for sample in samples:
            while not self.adc_ready:
                steps.append(self.step(None, adc_valid=False, aer_ready=True))
            steps.append(self.step(sample, adc_valid=True, aer_ready=True))
        while drain and self.aer_valid:
            steps.append(self.step(None, adc_valid=False, aer_ready=True))
        return steps


__all__ = ["ADCSpikeConfig", "ADCSpikeStep", "ADCToSpikeReference"]
