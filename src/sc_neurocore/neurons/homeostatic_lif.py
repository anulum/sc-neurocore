# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LIF neuron with homeostatic threshold adaptation

import math
from dataclasses import dataclass
from typing import Any, Dict

from ..constants import (
    HOMEOSTATIC_ADAPTATION_RATE,
    HOMEOSTATIC_TARGET_RATE,
    HOMEOSTATIC_THRESHOLD_CEILING_MULT,
    HOMEOSTATIC_THRESHOLD_FLOOR,
    HOMEOSTATIC_TRACE_DECAY,
)
from .stochastic_lif import StochasticLIFNeuron

THRESHOLD_FLOOR = HOMEOSTATIC_THRESHOLD_FLOOR
THRESHOLD_CEILING_MULT = HOMEOSTATIC_THRESHOLD_CEILING_MULT


@dataclass
class HomeostaticLIFNeuron(StochasticLIFNeuron):
    """
    LIF neuron with homeostatic threshold adaptation.

    Self-regulates firing rate toward a target setpoint via exponential
    moving average of spike rate. Based on Turrigiano (2012).

    Example
    -------
    >>> neuron = HomeostaticLIFNeuron(target_rate=0.1, noise_std=0.0)
    >>> for _ in range(200):
    ...     neuron.step(1.5)
    >>> neuron.v_threshold != 1.0  # threshold adapted
    True
    """

    target_rate: float = HOMEOSTATIC_TARGET_RATE
    adaptation_rate: float = HOMEOSTATIC_ADAPTATION_RATE
    rate_trace: float = 0.0
    trace_decay: float = HOMEOSTATIC_TRACE_DECAY

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.target_rate) or not 0.0 <= self.target_rate <= 1.0:
            raise ValueError("target_rate must be finite and within [0, 1]")
        if not math.isfinite(self.adaptation_rate) or self.adaptation_rate < 0.0:
            raise ValueError("adaptation_rate must be finite and non-negative")
        if not math.isfinite(self.rate_trace) or not 0.0 <= self.rate_trace <= 1.0:
            raise ValueError("rate_trace must be finite and within [0, 1]")
        if not math.isfinite(self.trace_decay) or not 0.0 <= self.trace_decay <= 1.0:
            raise ValueError("trace_decay must be finite and within [0, 1]")

        self.initial_threshold: float = self.v_threshold

    def step(self, input_current: float) -> int:
        spike = super().step(input_current)

        self.rate_trace = self.rate_trace * self.trace_decay + spike * (1.0 - self.trace_decay)

        error = self.rate_trace - self.target_rate
        self.v_threshold += self.adaptation_rate * error
        self.v_threshold = max(
            THRESHOLD_FLOOR,
            min(self.v_threshold, self.initial_threshold * THRESHOLD_CEILING_MULT),
        )

        return spike

    def get_state(self) -> Dict[str, Any]:
        s = super().get_state()
        s["threshold"] = float(self.v_threshold)
        s["rate_trace"] = float(self.rate_trace)
        return s
