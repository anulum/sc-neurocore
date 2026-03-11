# SPDX-License-Identifier: AGPL-3.0-or-later
from dataclasses import dataclass
from typing import Dict, Any
from .stochastic_lif import StochasticLIFNeuron

THRESHOLD_FLOOR = 0.1  # prevent threshold collapse to zero
THRESHOLD_CEILING_MULT = 10.0  # max threshold = initial * this factor


@dataclass
class HomeostaticLIFNeuron(StochasticLIFNeuron):
    """
    LIF Neuron with Homeostatic Threshold Adaptation.
    Self-regulates firing rate to a target setpoint.
    """

    target_rate: float = 0.1
    adaptation_rate: float = 0.01
    rate_trace: float = 0.0
    trace_decay: float = 0.95

    def __post_init__(self) -> None:
        super().__post_init__()
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
