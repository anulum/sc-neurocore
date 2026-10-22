# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-event layer for narrow links (threshold crossings and spike-band power)

"""Spike-event layer: threshold-crossing events and spike-band power for narrow links.

A Bluetooth-class link cannot carry broadband electrode data, but BCI decoders need far less:
which channels crossed a spike threshold in each short bin, and how much spike-band power each
channel carries. The encoder computes both from raw int16 samples and codes them with the
adaptive range coder of the streaming lossless coder:

1. Each channel passes a second-order Butterworth high-pass (default 250 Hz, Q14 coefficients).
2. The noise level is a streaming estimate of the median of ``|y|`` (steps shrinking from 1/32
   to 1/4096 across the warm-up); the threshold is ``threshold_sigma`` times
   ``median / 0.6745`` (at least one unit).
3. An event is a downward threshold crossing after the warm-up, at least ``refractory_samples``
   after the channel's previous event, so a channel has at most one event per bin.
4. Spike-band power is the mean ``|z|`` over ``power_bins`` bins, ``z`` being the high-passed
   signal through a second-order low-pass (default 1000 Hz), shifted right by ``power_shift``.

Every packet carries the bins completed by its window (one adaptive bit per channel and bin,
optionally the event's sample offset in the bin) and the completed band-power values. All
arithmetic is integer, so the Rust engine (``backend="rust"`` or ``"auto"``) and this module's
pure-Python reference (``"python"``) produce identical packets; the reference is far too slow
for real-time use.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from .waveform_stream import (
    _PROB_INIT,
    STREAM_BACKENDS,
    _RangeDecoder,
    _RangeEncoder,
    _ResidualModel,
    _sample_count,
    _unzigzag,
    _zigzag,
)

COEFFICIENT_BITS = 14
NOISE_BITS = 4
THRESHOLD_BITS = 12
WARMUP_STEP_SHIFT = 5
NOISE_STEP_SHIFT = 12
MAX_WARMUP_SAMPLES = 1 << 40
MAX_BIN_SAMPLES = 1024
EVENT_ADAPT_SHIFT = 4
_EVENT_PROB_INIT = 1 << 15
SIGNAL_LIMIT = 1 << 24
MAD_TO_SIGMA = 0.6745
_PACKET = "corrupted spike-event packet"


def _q14(value: float) -> int:
    return math.floor(value * float(1 << COEFFICIENT_BITS) + 0.5)


def _butterworth(cutoff_hz: float, rate_hz: float, highpass: bool) -> tuple[int, ...]:
    if (
        not (math.isfinite(cutoff_hz) and math.isfinite(rate_hz))
        or cutoff_hz < rate_hz / 1000.0
        or cutoff_hz >= 0.45 * rate_hz
    ):
        raise ValueError(
            f"filter cutoff {cutoff_hz} Hz must lie in [rate/1000, 0.45 * rate) "
            f"for rate {rate_hz} Hz"
        )
    k = math.tan(math.pi * cutoff_hz / rate_hz)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    if highpass:
        b0, b1, b2 = norm, -2.0 * norm, norm
    else:
        b0 = k * k * norm
        b1, b2 = 2.0 * b0, b0
    a1 = _q14(2.0 * (k * k - 1.0) * norm)
    a2 = _q14((1.0 - math.sqrt(2.0) * k + k * k) * norm)
    one = 1 << COEFFICIENT_BITS
    if _q14(b0) == 0 or abs(a2) >= one or abs(a1) >= one + a2:
        raise ValueError(
            f"filter cutoff {cutoff_hz} Hz at {rate_hz} Hz cannot be represented in Q14"
        )
    return (_q14(b0), _q14(b1), _q14(b2), a1, a2)


def highpass_coefficients(cutoff_hz: float, rate_hz: float) -> tuple[int, ...]:
    """Return the Q14 coefficients ``(b0, b1, b2, a1, a2)`` of the second-order Butterworth high-pass.

    Raises
    ------
    ValueError
        If the cutoff is outside ``[rate / 1000, 0.45 * rate)`` or the quantised filter is unstable.
    """
    return _butterworth(float(cutoff_hz), float(rate_hz), True)


def lowpass_coefficients(cutoff_hz: float, rate_hz: float) -> tuple[int, ...]:
    """Return the Q14 coefficients ``(b0, b1, b2, a1, a2)`` of the second-order Butterworth low-pass.

    Raises
    ------
    ValueError
        If the cutoff is outside ``[rate / 1000, 0.45 * rate)`` or the quantised filter is unstable.
    """
    return _butterworth(float(cutoff_hz), float(rate_hz), False)


def threshold_factor(threshold_sigma: float) -> int:
    """Return ``threshold_sigma / 0.6745`` in Q12, the factor applied to the median ``|y|``."""
    return math.floor(threshold_sigma / MAD_TO_SIGMA * float(1 << THRESHOLD_BITS) + 0.5)


def _offset_bits(bin_samples: int) -> int:
    return (bin_samples - 1).bit_length()


class _Biquad:
    """Direct-form-I biquad with Q14 coefficients and rounded, clamped output."""

    def __init__(self, coefficients: tuple[int, ...]) -> None:
        self.b0, self.b1, self.b2, self.a1, self.a2 = coefficients
        self.x1 = self.x2 = self.y1 = self.y2 = 0

    def step(self, x: int) -> int:
        acc = (
            self.b0 * x
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        y = min(
            max((acc + (1 << (COEFFICIENT_BITS - 1))) >> COEFFICIENT_BITS, -SIGNAL_LIMIT),
            SIGNAL_LIMIT,
        )
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y


class _EventModel:
    """Per-channel coding state shared by encoder and decoder."""

    def __init__(self) -> None:
        self.probs = [_EVENT_PROB_INIT]
        self.previous_power = 0


class _GroupModel:
    """Per-group coding state: channel models, offset tree and band-power model."""

    def __init__(self, start: int, end: int, bin_samples: int) -> None:
        self.start, self.end = start, end
        self.channels = [_EventModel() for _ in range(end - start)]
        self.offsets = [_PROB_INIT] * (1 << _offset_bits(bin_samples))
        self.power = _ResidualModel()


class _Detector:
    """Per-channel detector state (encoder only)."""

    def __init__(
        self, highpass: tuple[int, ...], lowpass: tuple[int, ...], refractory: int
    ) -> None:
        self.highpass = _Biquad(highpass)
        self.lowpass = _Biquad(lowpass)
        self.noise = 0
        self.below = False
        self.since_event = refractory
        self.power_sum = 0
        self.carried: int | None = None


def _groups(channels: int, groups: int, bin_samples: int) -> list[_GroupModel]:
    return [
        _GroupModel(g * channels // groups, (g + 1) * channels // groups, bin_samples)
        for g in range(groups)
    ]


def _validate_layout(channels: int, bin_samples: int, power_bins: int, groups: int) -> None:
    if channels < 1:
        raise ValueError("a spike-event stream needs at least one channel")
    if not 1 <= bin_samples <= MAX_BIN_SAMPLES:
        raise ValueError(f"bin_samples must be 1..{MAX_BIN_SAMPLES}")
    if not 0 <= power_bins <= 65535:
        raise ValueError("power_bins must be 0..65535")
    if not 1 <= groups <= min(channels, 65535):
        raise ValueError("groups must be 1..min(channels, 65535)")


def _validate_settings(settings: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Check every encoder setting; return the high-pass and low-pass coefficients."""
    _validate_layout(
        settings["channels"], settings["bin_samples"], settings["power_bins"], settings["groups"]
    )
    if not settings["bin_samples"] <= settings["refractory_samples"] <= 1 << 20:
        raise ValueError("refractory_samples must be bin_samples..2^20")
    sigma = settings["threshold_sigma"]
    if not (math.isfinite(sigma) and 0.0 < sigma <= 100.0):
        raise ValueError("threshold_sigma must be in (0, 100]")
    if not 0 <= settings["warmup_samples"] <= MAX_WARMUP_SAMPLES:
        raise ValueError("warmup_samples must be 0..2^40")
    if not 0 <= settings["power_shift"] <= 15:
        raise ValueError("power_shift must be 0..15")
    rate = settings["sample_rate_hz"]
    return (
        highpass_coefficients(settings["highpass_hz"], rate),
        lowpass_coefficients(settings["lowpass_hz"], rate),
    )


def _engine(backend: str) -> ModuleType | None:
    """Return the engine module with the spike-event layer, or None for the Python reference."""
    if backend not in STREAM_BACKENDS:
        raise ValueError(f"backend must be one of {STREAM_BACKENDS}, got {backend!r}")
    if backend == "python":
        return None
    try:
        engine: ModuleType | None = importlib.import_module(
            "sc_neurocore_engine.sc_neurocore_engine"
        )
    except ImportError:
        engine = None
    if engine is not None and hasattr(engine, "SpikeEventEncoder"):
        return engine
    if backend == "rust":
        raise ValueError("backend 'rust' needs sc_neurocore_engine with the spike-event layer")
    return None


class _PyEncoder:
    """Pure-Python reference encoder (settings checked by ``_validate_settings``)."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.s = settings
        highpass, lowpass = _validate_settings(settings)
        self.factor = threshold_factor(settings["threshold_sigma"])
        self.groups = _groups(settings["channels"], settings["groups"], settings["bin_samples"])
        self.detectors = [
            _Detector(highpass, lowpass, settings["refractory_samples"])
            for _ in range(settings["channels"])
        ]
        self.samples_seen = 0

    def encode(self, window: np.ndarray[Any, Any]) -> bytes:
        s = self.s
        samples = window.shape[0]
        columns = window.astype(np.int64).T.tolist()
        start = self.samples_seen
        bin_samples = s["bin_samples"]
        period = s["power_bins"] * bin_samples
        first_bin, end_bin = start // bin_samples, (start + samples) // bin_samples
        warmup, refractory = s["warmup_samples"], s["refractory_samples"]
        packet = bytearray()
        for group in self.groups:
            encoder = _RangeEncoder()
            for offset, channel in enumerate(group.channels):
                index = group.start + offset
                detector = self.detectors[index]
                events = [] if detector.carried is None else [detector.carried]
                detector.carried = None
                powers = []
                for row, x in enumerate(columns[index]):
                    t = start + row
                    y = detector.highpass.step(x)
                    threshold = max(
                        (detector.noise * self.factor) >> THRESHOLD_BITS, 1 << NOISE_BITS
                    )
                    below = (y << NOISE_BITS) < -threshold
                    if (
                        below
                        and not detector.below
                        and detector.since_event >= refractory
                        and t >= warmup
                    ):
                        events.append(t)
                        detector.since_event = 0
                    detector.below = below
                    detector.since_event += 1
                    magnitude = abs(y) << NOISE_BITS
                    shift = _noise_step_shift(t, warmup)
                    step = max(detector.noise >> shift, 1)
                    if magnitude > detector.noise:
                        detector.noise += step
                    elif magnitude < detector.noise:
                        detector.noise -= step
                    if period:
                        detector.power_sum += abs(detector.lowpass.step(y))
                        if (t + 1) % period == 0:
                            powers.append((detector.power_sum // period) >> s["power_shift"])
                            detector.power_sum = 0
                pending = iter(events)
                event = next(pending, None)
                for bin_index in range(first_bin, end_bin):
                    hit = event is not None and event // bin_samples == bin_index
                    encoder.bit_precise(channel.probs, 0, int(hit), EVENT_ADAPT_SHIFT)
                    if hit:
                        if s["sample_timing"]:
                            _put_offset(encoder, group.offsets, event % bin_samples, bin_samples)
                        event = next(pending, None)
                detector.carried = event
                for value in powers:
                    group.power.put(encoder, _zigzag(value - channel.previous_power))
                    channel.previous_power = value
            body = encoder.finish()
            packet += len(body).to_bytes(4, "big") + body
        self.samples_seen += samples
        return bytes(packet)


def _noise_step_shift(t: int, warmup: int) -> int:
    """Return the step shift: 5 at the start of the warm-up, rising to 12 at its end and after."""
    if t >= warmup:
        return NOISE_STEP_SHIFT
    ramp = t * (NOISE_STEP_SHIFT - WARMUP_STEP_SHIFT + 1) // warmup
    return min(WARMUP_STEP_SHIFT + ramp, NOISE_STEP_SHIFT)


def _put_offset(encoder: _RangeEncoder, tree: list[int], offset: int, bin_samples: int) -> None:
    node = 1
    for i in range(_offset_bits(bin_samples) - 1, -1, -1):
        bit = (offset >> i) & 1
        encoder.bit(tree, node, bit)
        node = node * 2 + bit


@dataclass(frozen=True)
class SpikeEventWindow:
    """Events and spike-band power decoded from one packet.

    Attributes
    ----------
    first_bin : int
        Absolute index of the first bin in ``events``.
    events : ndarray
        uint8 ``(bins, channels)``: 1 where the channel has an event in the bin.
    offsets : ndarray or None
        int16 ``(bins, channels)`` sample offsets inside the bin, -1 without an event; None
        unless the stream uses ``timing="sample"``.
    first_period : int
        Absolute index of the first band-power period in ``power``.
    power : ndarray
        int32 ``(periods, channels)`` spike-band power values.
    """

    first_bin: int
    events: np.ndarray[Any, Any]
    offsets: np.ndarray[Any, Any] | None
    first_period: int
    power: np.ndarray[Any, Any]


class _PyDecoder:
    """Pure-Python reference decoder."""

    def __init__(
        self, channels: int, bin_samples: int, sample_timing: bool, power_bins: int, groups: int
    ) -> None:
        _validate_layout(channels, bin_samples, power_bins, groups)
        self.channels, self.bin_samples = channels, bin_samples
        self.sample_timing, self.power_bins = sample_timing, power_bins
        self.groups = _groups(channels, groups, bin_samples)
        self.samples_seen = 0
        self.broken = False

    def decode(
        self, packet: bytes, samples: int
    ) -> tuple[int, np.ndarray[Any, Any], np.ndarray[Any, Any] | None, int, np.ndarray[Any, Any]]:
        if self.broken:
            raise ValueError(
                "spike-event decoder is broken by an earlier corrupted packet; start a new decoder"
            )
        bodies = []
        pos = 0
        for _ in self.groups:
            if pos + 4 > len(packet):
                raise ValueError(f"{_PACKET}: missing group length")
            length = int.from_bytes(packet[pos : pos + 4], "big")
            if pos + 4 + length > len(packet):
                raise ValueError(f"{_PACKET}: group longer than packet")
            body = packet[pos + 4 : pos + 4 + length]
            if len(body) < 5 or body[0] != 0:
                raise ValueError(f"{_PACKET}: bad range-coder start")
            bodies.append(body)
            pos += 4 + length
        if pos != len(packet):
            raise ValueError(f"{_PACKET}: trailing bytes")
        end = self.samples_seen + samples
        period = self.power_bins * self.bin_samples
        bins = end // self.bin_samples - self.samples_seen // self.bin_samples
        periods = end // period - self.samples_seen // period if period else 0
        for group, body in zip(self.groups, bodies, strict=True):
            width, room = group.end - group.start, len(body) + 8
            if end >= 1 << 64 or bins * width > 32768 * room or periods * width > 64 * room:
                raise ValueError(f"{_PACKET}: more samples requested than the packet can hold")
        try:
            result = self._decode_groups(bodies, samples)
        except ValueError:
            self.broken = True
            raise
        self.samples_seen += samples
        return result

    def _decode_groups(
        self, bodies: list[bytes], samples: int
    ) -> tuple[int, np.ndarray[Any, Any], np.ndarray[Any, Any] | None, int, np.ndarray[Any, Any]]:
        bin_samples = self.bin_samples
        period = self.power_bins * bin_samples
        end = self.samples_seen + samples
        first_bin = self.samples_seen // bin_samples
        bins = end // bin_samples - first_bin
        first_period = self.samples_seen // period if period else 0
        periods = end // period - first_period if period else 0
        events = np.zeros((bins, self.channels), dtype=np.uint8)
        offsets = np.full((bins, self.channels), -1, dtype=np.int16) if self.sample_timing else None
        power = np.zeros((periods, self.channels), dtype=np.int32)
        bits = _offset_bits(bin_samples)
        for group, body in zip(self.groups, bodies, strict=True):
            decoder = _RangeDecoder(body)
            try:
                for offset, channel in enumerate(group.channels):
                    column = group.start + offset
                    for row in range(bins):
                        event = decoder.bit_precise(channel.probs, 0, EVENT_ADAPT_SHIFT)
                        events[row, column] = event
                        if event and offsets is not None:
                            node = 1
                            for _ in range(bits):
                                node = node * 2 + decoder.bit(group.offsets, node)
                            position = node - (1 << bits)
                            if position >= bin_samples:
                                raise ValueError(f"{_PACKET}: event offset beyond its bin")
                            offsets[row, column] = position
                    for row in range(periods):
                        u = group.power.get(decoder)
                        if u is None:
                            raise ValueError(f"{_PACKET}: unexpected marker in band power")
                        value = channel.previous_power + _unzigzag(u)
                        if not 0 <= value <= SIGNAL_LIMIT:
                            raise ValueError(f"{_PACKET}: band power out of range")
                        channel.previous_power = value
                        power[row, column] = value
            except ValueError as error:
                # The shared range coder words its errors for the lossless stream.
                for shared in ("range coder ran past its bytes", "escape too long"):
                    if shared in str(error):
                        raise ValueError(f"{_PACKET}: {shared}") from None
                raise
            if decoder.pos != len(decoder.data):
                raise ValueError(f"{_PACKET}: trailing range-coder bytes")
        return first_bin, events, offsets, first_period, power


_TIMINGS = ("bin", "sample")


def _window(window: np.ndarray[Any, Any], channels: int) -> np.ndarray[Any, Any]:
    array = np.asarray(window)
    if array.ndim != 2 or array.shape[1] != channels:
        raise ValueError(f"window must be (samples, {channels})")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        if not bool(np.all(np.isfinite(array))) or not bool(np.all(array == np.round(array))):
            raise ValueError("spike-event coding needs integer samples")
    if array.size and (array.min() < -32768 or array.max() > 32767):
        raise ValueError("spike-event coding needs samples in the int16 range")
    return np.ascontiguousarray(array, dtype=np.int16)


class SpikeEventEncoder:
    """Stateful encoder of threshold-crossing events and spike-band power.

    Parameters
    ----------
    channels : int
        Channels per window.
    sample_rate_hz : float
        Sampling rate of the raw stream.
    bin_samples : int or None
        Samples per event bin (1..1024); None gives about 1 ms.
    refractory_samples : int or None
        Minimum distance between two events of a channel (at least ``bin_samples``); None
        equals ``bin_samples``.
    threshold_sigma : float
        Threshold in noise standard deviations (MAD estimate), in (0, 100].
    highpass_hz : float
        Cutoff of the detection high-pass.
    lowpass_hz : float
        Upper cutoff of the spike band for the band power.
    warmup_samples : int or None
        Samples before the first event may be reported (the noise estimate settles); None
        gives about 100 ms.
    timing : str
        ``"bin"`` sends which bins hold an event; ``"sample"`` adds each event's sample
        offset inside its bin.
    power_bins : int
        Bins per band-power value (0 disables band power, at most 65535).
    power_shift : int
        Right shift of the band-power values (0..15; e.g. 6 for a 10-bit ADC scaled to int16).
    groups : int
        Contiguous channel groups, each with its own range coder (coded in parallel in Rust).
    backend : str
        ``"auto"`` (Rust when installed), ``"rust"`` or ``"python"``; identical packets.
    """

    def __init__(
        self,
        channels: int,
        sample_rate_hz: float,
        *,
        bin_samples: int | None = None,
        refractory_samples: int | None = None,
        threshold_sigma: float = 4.5,
        highpass_hz: float = 250.0,
        lowpass_hz: float = 1000.0,
        warmup_samples: int | None = None,
        timing: str = "bin",
        power_bins: int = 20,
        power_shift: int = 0,
        groups: int = 1,
        backend: str = "auto",
    ) -> None:
        if timing not in _TIMINGS:
            raise ValueError(f"timing must be one of {_TIMINGS}, got {timing!r}")
        rate = float(sample_rate_hz)
        if not (math.isfinite(rate) and rate > 0.0):
            raise ValueError("sample_rate_hz must be positive")
        bins = max(round(rate / 1000.0), 1) if bin_samples is None else int(bin_samples)
        settings: dict[str, Any] = {
            "channels": int(channels),
            "sample_rate_hz": rate,
            "bin_samples": bins,
            "refractory_samples": bins if refractory_samples is None else int(refractory_samples),
            "threshold_sigma": float(threshold_sigma),
            "highpass_hz": float(highpass_hz),
            "lowpass_hz": float(lowpass_hz),
            "warmup_samples": round(rate / 10.0) if warmup_samples is None else int(warmup_samples),
            "sample_timing": timing == "sample",
            "power_bins": int(power_bins),
            "power_shift": int(power_shift),
            "groups": int(groups),
        }
        _validate_settings(settings)
        self.channels = settings["channels"]
        self.settings = dict(settings)
        native = _engine(backend)
        self._coder: Any = (
            native.SpikeEventEncoder(*settings.values())
            if native is not None
            else _PyEncoder(settings)
        )

    def encode(self, window: np.ndarray[Any, Any]) -> bytes:
        """Encode a ``(samples, channels)`` window of int16-range integers into a packet.

        Raises
        ------
        ValueError
            For a malformed window.
        """
        return bytes(self._coder.encode(_window(window, self.channels)))


class SpikeEventDecoder:
    """Stateful decoder matching ``SpikeEventEncoder``.

    Parameters
    ----------
    channels, bin_samples, power_bins, groups : int
        As given to (or derived by) the encoder; ``encoder.settings`` holds the values.
    timing : str
        ``"bin"`` or ``"sample"``, as for the encoder.
    backend : str
        ``"auto"``, ``"rust"`` or ``"python"``.
    """

    def __init__(
        self,
        channels: int,
        bin_samples: int,
        *,
        timing: str = "bin",
        power_bins: int = 20,
        groups: int = 1,
        backend: str = "auto",
    ) -> None:
        if timing not in _TIMINGS:
            raise ValueError(f"timing must be one of {_TIMINGS}, got {timing!r}")
        args = (int(channels), int(bin_samples), timing == "sample", int(power_bins), int(groups))
        _validate_layout(args[0], args[1], args[3], args[4])
        native = _engine(backend)
        self._coder: Any = (
            native.SpikeEventDecoder(*args) if native is not None else _PyDecoder(*args)
        )

    def decode(self, packet: bytes, samples: int) -> SpikeEventWindow:
        """Decode the packet of a window of ``samples`` samples.

        Raises
        ------
        ValueError
            For a malformed packet. Framing errors leave the decoder unchanged; content found
            invalid while decoding breaks it, and every later packet is refused. A sample count
            that is not an integer in ``[0, 2**64)`` is refused before decoding.
        """
        count = _sample_count(samples)
        first_bin, events, offsets, first_period, power = self._coder.decode(bytes(packet), count)
        return SpikeEventWindow(
            int(first_bin),
            np.asarray(events, dtype=np.uint8),
            None if offsets is None else np.asarray(offsets, dtype=np.int16),
            int(first_period),
            np.asarray(power, dtype=np.int32),
        )
