# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Input encoders that declare exactly what they do

"""Encoders whose declaration is enough to rebuild them.

How events or values become the spike tensor a network sees — the time step,
the window, what happens to an event after it, whether polarities share a
channel — changes every result downstream, and it is usually lost in a
training script. Each encoder here is a frozen value: :meth:`declaration`
returns its full description as JSON, :func:`encoder_from_declaration`
rebuilds the identical encoder from it, and the declaration's digest can be
recorded with a run.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .encoding import latency_encode, poisson_encode

ENCODER_SCHEMA = "sc-neurocore.input-encoder.v1"


def _digest(declaration: dict[str, Any]) -> str:
    canonical = json.dumps(declaration, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _positive_finite(name: str, value: float) -> None:
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be positive and finite; got {value!r}")


@dataclass(frozen=True, slots=True)
class EventBinning:
    """Bin camera events into a binary spike tensor.

    An event at ``t`` ms lands in step ``floor(t / dt_ms)``; events at or after
    ``n_steps * dt_ms`` are dropped, never merged into the last step. With
    ``polarity="separate"`` ON and OFF events have their own channels (OFF
    first), with ``"merge"`` they share one. An event outside the sensor or
    before time zero is refused rather than clipped.

    Attributes
    ----------
    dt_ms:
        Step length in milliseconds.
    n_steps:
        Number of steps in the window.
    width, height:
        Sensor geometry in pixels.
    polarity:
        ``"separate"`` or ``"merge"``.
    """

    dt_ms: float
    n_steps: int
    width: int
    height: int
    polarity: Literal["separate", "merge"] = "separate"

    def __post_init__(self) -> None:
        """Refuse a setting the declaration could not state faithfully."""
        _positive_finite("dt_ms", self.dt_ms)
        _positive_int("n_steps", self.n_steps)
        _positive_int("width", self.width)
        _positive_int("height", self.height)
        if self.polarity not in ("separate", "merge"):
            raise ValueError(f"polarity must be 'separate' or 'merge'; got {self.polarity!r}")

    @property
    def channels(self) -> int:
        """Channels per step: pixels, twice over when polarities are separate."""
        pixels = self.width * self.height
        return 2 * pixels if self.polarity == "separate" else pixels

    def declaration(self) -> dict[str, Any]:
        """Return the full description, enough to rebuild this encoder."""
        return {
            "schema": ENCODER_SCHEMA,
            "encoder": "event-binning",
            "dt_ms": self.dt_ms,
            "n_steps": self.n_steps,
            "width": self.width,
            "height": self.height,
            "polarity": self.polarity,
            "step": "floor(t_ms / dt_ms)",
            "late_events": "dropped",
            "channel": "polarity * width * height + y * width + x"
            if self.polarity == "separate"
            else "y * width + x",
            "output": "bool (n_steps, channels)",
        }

    @property
    def digest(self) -> str:
        """``sha256:`` over the declaration."""
        return _digest(self.declaration())

    def encode(self, events: npt.ArrayLike) -> np.ndarray[Any, Any]:
        """Bin ``(N, 4)`` events with columns ``x, y, polarity, t_ms``.

        Parameters
        ----------
        events:
            Events as the event loaders return them.

        Returns
        -------
        numpy.ndarray
            ``bool`` array of shape ``(n_steps, channels)``.

        Raises
        ------
        ValueError
            On a malformed array, an event outside the sensor, a polarity
            other than 0 or 1, or a negative or non-finite time.
        """
        array = np.asarray(events, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError(f"events must have shape (N, 4); got {array.shape}")
        spikes = np.zeros((self.n_steps, self.channels), dtype=bool)
        if array.shape[0] == 0:
            return spikes
        x, y, polarity, time_ms = array.T
        if not np.all(np.isfinite(array)):
            raise ValueError("events hold a non-finite value")
        if np.any(x != np.floor(x)) or np.any(y != np.floor(y)):
            raise ValueError("pixel addresses must be whole numbers")
        if np.any((x < 0) | (x >= self.width) | (y < 0) | (y >= self.height)):
            raise ValueError(f"an event lies outside the {self.width} x {self.height} sensor")
        if np.any((polarity != 0) & (polarity != 1)):
            raise ValueError("polarity must be 0 or 1")
        if np.any(time_ms < 0):
            raise ValueError("an event has a negative time")
        step = np.floor(time_ms / self.dt_ms).astype(np.int64)
        inside = step < self.n_steps
        channel = y.astype(np.int64) * self.width + x.astype(np.int64)
        if self.polarity == "separate":
            channel = channel + polarity.astype(np.int64) * self.width * self.height
        spikes[step[inside], channel[inside]] = True
        return spikes


@dataclass(frozen=True, slots=True)
class PoissonRates:
    """Encode per-step firing probabilities as seeded Bernoulli spike trains.

    Attributes
    ----------
    n_steps:
        Number of steps.
    dt_ms:
        Step length; the probability per step is ``rate * dt_ms``, clipped
        to ``[0, 1]``.
    seed:
        Generator seed; the same seed and input give the same spikes.
    """

    n_steps: int
    dt_ms: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Refuse a setting the declaration could not state faithfully."""
        _positive_int("n_steps", self.n_steps)
        _positive_finite("dt_ms", self.dt_ms)

    def declaration(self) -> dict[str, Any]:
        """Return the full description, enough to rebuild this encoder."""
        return {
            "schema": ENCODER_SCHEMA,
            "encoder": "poisson-rates",
            "n_steps": self.n_steps,
            "dt_ms": self.dt_ms,
            "seed": self.seed,
            "generator": "numpy.random.default_rng(seed).random((n_steps, N))",
            "probability": "clip(rate * dt_ms, 0, 1)",
            "output": "bool (n_steps, N)",
        }

    @property
    def digest(self) -> str:
        """``sha256:`` over the declaration."""
        return _digest(self.declaration())

    def encode(self, rates: npt.ArrayLike) -> np.ndarray[Any, Any]:
        """Return the spike trains for a vector of rates."""
        return poisson_encode(rates, self.n_steps, dt_ms=self.dt_ms, seed=self.seed)


@dataclass(frozen=True, slots=True)
class FirstSpikeLatency:
    """Encode values in ``[0, 1]`` as one spike each, larger values earlier.

    Attributes
    ----------
    n_steps:
        Number of steps.
    tau:
        The spike of value ``v`` falls in step ``int(tau * (1 - v))``,
        limited to the window.
    """

    n_steps: int
    tau: float = 5.0

    def __post_init__(self) -> None:
        """Refuse a setting the declaration could not state faithfully."""
        _positive_int("n_steps", self.n_steps)
        _positive_finite("tau", self.tau)

    def declaration(self) -> dict[str, Any]:
        """Return the full description, enough to rebuild this encoder."""
        return {
            "schema": ENCODER_SCHEMA,
            "encoder": "first-spike-latency",
            "n_steps": self.n_steps,
            "tau": self.tau,
            "step": "int(clip(tau * (1 - v), 0, n_steps - 1))",
            "values_outside_unit_interval": "refused",
            "output": "bool (n_steps, N)",
        }

    @property
    def digest(self) -> str:
        """``sha256:`` over the declaration."""
        return _digest(self.declaration())

    def encode(self, values: npt.ArrayLike) -> np.ndarray[Any, Any]:
        """Return one spike per value; a value outside ``[0, 1]`` is refused."""
        return latency_encode(values, self.n_steps, tau=self.tau, strict=True)


InputEncoder = EventBinning | PoissonRates | FirstSpikeLatency


def encoder_from_declaration(declaration: dict[str, Any]) -> InputEncoder:
    """Rebuild the encoder a declaration describes.

    Parameters
    ----------
    declaration:
        A declaration as :meth:`EventBinning.declaration` and its siblings
        return it.

    Returns
    -------
    EventBinning or PoissonRates or FirstSpikeLatency
        The encoder; its own declaration equals the one given.

    Raises
    ------
    ValueError
        On another schema, an unknown encoder, or a declaration that is not
        exactly what the rebuilt encoder declares.
    """
    if declaration.get("schema") != ENCODER_SCHEMA:
        raise ValueError(f"encoder schema {declaration.get('schema')!r} is not {ENCODER_SCHEMA!r}")
    kind = declaration.get("encoder")
    encoder: InputEncoder
    if kind == "event-binning":
        encoder = EventBinning(
            dt_ms=float(declaration["dt_ms"]),
            n_steps=int(declaration["n_steps"]),
            width=int(declaration["width"]),
            height=int(declaration["height"]),
            polarity=declaration["polarity"],
        )
    elif kind == "poisson-rates":
        encoder = PoissonRates(
            n_steps=int(declaration["n_steps"]),
            dt_ms=float(declaration["dt_ms"]),
            seed=int(declaration["seed"]),
        )
    elif kind == "first-spike-latency":
        encoder = FirstSpikeLatency(
            n_steps=int(declaration["n_steps"]), tau=float(declaration["tau"])
        )
    else:
        raise ValueError(f"unknown encoder {kind!r}")
    if encoder.declaration() != declaration:
        raise ValueError(
            "the declaration does not match what this version of the encoder does; "
            "it cannot be rebuilt faithfully"
        )
    return encoder


__all__ = [
    "ENCODER_SCHEMA",
    "EventBinning",
    "FirstSpikeLatency",
    "InputEncoder",
    "PoissonRates",
    "encoder_from_declaration",
]
