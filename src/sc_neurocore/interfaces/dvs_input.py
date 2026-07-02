# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Interface for Dynamic Vision Sensors (Event Cameras)

"""Dynamic Vision Sensor ingress layer for event-camera AER streams."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt

DVSEvent: TypeAlias = tuple[int, int, float, int]
Float32Array: TypeAlias = npt.NDArray[np.float32]
UInt8Array: TypeAlias = npt.NDArray[np.uint8]


def _positive_integer(value: object, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(message)
    integer = int(value)
    if integer <= 0:
        raise ValueError(message)
    return integer


def _positive_finite_float(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(message)
    return scalar


def _probability_surface(surface: Float32Array) -> Float32Array:
    return cast(Float32Array, np.tanh(surface).astype(np.float32, copy=False))


@dataclass
class DVSInputLayer:
    """Convert Dynamic Vision Sensor AER events into stochastic bitstreams.

    Parameters
    ----------
    height:
        Positive number of pixel rows in the event-camera frame.
    width:
        Positive number of pixel columns in the event-camera frame.
    decay_tau:
        Positive finite exponential-decay time constant in milliseconds.
    """

    height: int
    width: int
    decay_tau: float = 100.0  # Time constant to decay old events
    surface: Float32Array = field(init=False, repr=False)
    last_update_time: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        """Validate sensor geometry and allocate the internal event surface."""
        self.height = _positive_integer(self.height, "height and width must be positive integers")
        self.width = _positive_integer(self.width, "height and width must be positive integers")
        self.decay_tau = _positive_finite_float(
            self.decay_tau,
            "decay_tau must be finite and positive",
        )
        # Surface potential representing event density
        self.surface = cast(
            Float32Array,
            np.zeros((self.height, self.width), dtype=np.float32),
        )
        self.last_update_time = 0.0

    def process_events(self, events: list[DVSEvent]) -> Float32Array:
        """Integrate a timestamp-ordered batch of DVS events.

        Parameters
        ----------
        events:
            AER events encoded as ``(x, y, timestamp_ms, polarity)`` tuples.
            Coordinates outside the configured frame are ignored; malformed
            addresses, non-finite timestamps, timestamp rewinds, and invalid
            polarities are rejected before state mutation.

        Returns
        -------
        numpy.ndarray
            ``height x width`` probability frame in ``[0, 1]``.

        Raises
        ------
        ValueError
            If an event has a malformed address, non-finite or non-monotonic
            timestamp, timestamp earlier than the layer clock, or unsupported
            polarity.
        """
        if not events:
            return _probability_surface(self.surface)

        current_time = self._validate_events(events)
        dt = current_time - self.last_update_time

        # Exponential decay of old activity
        # V_new = V_old * exp(-dt/tau)
        decay_factor = float(np.exp(-dt / self.decay_tau))
        self.surface *= decay_factor

        # Add new events
        for x, y, _timestamp, _polarity in events:
            xi = int(x)
            yi = int(y)
            if 0 <= xi < self.width and 0 <= yi < self.height:
                # Polarity is usually -1 or 1.
                # We want activity map. Let's just accumulate magnitude or positive density.
                self.surface[yi, xi] += 1.0

        # Clip/Sigmoid to [0, 1] for SC generation
        # Simple saturation
        output_probs = _probability_surface(self.surface)  # Maps 0->0, High->1

        self.last_update_time = current_time
        return output_probs

    def _validate_events(self, events: list[DVSEvent]) -> float:
        previous_t: float | None = None
        latest_t = self.last_update_time
        for x, y, t, p in events:
            if isinstance(x, bool) or not isinstance(x, Integral):
                raise ValueError("event coordinates must be integer pixel addresses")
            if isinstance(y, bool) or not isinstance(y, Integral):
                raise ValueError("event coordinates must be integer pixel addresses")
            if isinstance(t, bool) or not isinstance(t, Real):
                raise ValueError("event timestamp must be finite")
            timestamp = float(t)
            if not math.isfinite(timestamp):
                raise ValueError("event timestamp must be finite")
            if previous_t is not None and timestamp < previous_t:
                raise ValueError("event timestamps must be monotonically non-decreasing")
            if timestamp < self.last_update_time:
                raise ValueError("event timestamp cannot be earlier than last update time")
            if isinstance(p, bool) or not isinstance(p, Integral) or int(p) not in {-1, 0, 1}:
                raise ValueError("event polarity must be -1, 0, or 1")
            previous_t = timestamp
            latest_t = timestamp
        return latest_t

    def generate_bitstream_frame(self, length: int = 256) -> UInt8Array:
        """Generate a stochastic bitstream cube from the current DVS surface.

        Parameters
        ----------
        length:
            Positive number of stochastic samples to generate per pixel.

        Returns
        -------
        numpy.ndarray
            ``height x width x length`` uint8 tensor containing only 0 and 1.

        Raises
        ------
        ValueError
            If ``length`` is not a positive integer.
        """
        bitstream_length = _positive_integer(length, "length must be a positive integer")
        probs = _probability_surface(self.surface)
        # Vectorized generation
        # (H, W, Length)
        rands = np.random.random((self.height, self.width, bitstream_length))
        bits = (rands < probs[:, :, None]).astype(np.uint8)
        return bits
