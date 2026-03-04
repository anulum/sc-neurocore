from __future__ import annotations
from typing import Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class FSMActivation:
    """
    Base class for FSM-based stochastic activation functions.

    The FSM takes a bitstream input and transitions between states.
    The output bit is determined by the current state (e.g., if state > N/2, out=1).
    This implements saturating non-linearities like Tanh or Sigmoid efficiently.
    """

    num_states: int
    initial_state: int

    def __post_init__(self):  # type: ignore
        self.state = self.initial_state

    def step(self, bit: int) -> int:
        raise NotImplementedError

    def process(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        output = np.zeros_like(bitstream)
        for i, bit in enumerate(bitstream):
            output[i] = self.step(bit)
        return output


@dataclass
class TanhFSM(FSMActivation):
    """
    Implements a Tanh-like function using a linear FSM.

    States: 0 to N-1
    Input 0: state -> max(0, state - 1)
    Input 1: state -> min(N-1, state + 1)
    Output: 1 if state >= N/2 else 0
    """

    def __init__(self, states: int = 16):
        self.num_states = states
        self.initial_state = states // 2
        super().__post_init__()  # type: ignore

    def step(self, bit: int) -> int:
        if bit == 1:
            if self.state < self.num_states - 1:
                self.state += 1
        else:
            if self.state > 0:
                self.state -= 1

        return 1 if self.state >= (self.num_states // 2) else 0


@dataclass
class ReLKFSM(FSMActivation):
    """
    Implements a Rectified Linear (ReLU-like) behavior.
    Can be complex in SC, often approximated or used with bipolar coding.
    Here we implement a simple saturating counter.
    """

    def __init__(self, states: int = 16):
        self.num_states = states
        self.initial_state = 0  # Start at 0
        super().__post_init__()  # type: ignore

    def step(self, bit: int) -> int:
        if bit == 1:
            if self.state < self.num_states - 1:
                self.state += 1
        else:
            if self.state > 0:
                self.state -= 1

        # Probabilistic output based on state?
        # Or threshold? ReLK usually implies simple pass-through if > 0.
        # This implementation is a "Stochastic Integrator"
        return 1 if self.state > 0 else 0
