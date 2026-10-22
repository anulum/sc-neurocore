# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Streaming lossless coder for multi-channel BCI electrode data

"""Streaming lossless coder for multi-channel int16 electrode data (any BCI system).

The coder keeps state across windows, so prediction and probabilities carry on from packet to
packet, and every packet decodes to exactly the samples that were encoded. Each packet adds
about 9 bytes per channel group (a 4-byte length and the range-coder flush), so short packets
cost little only when many channels share a group (1024 channels in one group: 3.4646x with
1 ms windows against 3.4668x with 10 ms; one channel in 1 ms packets falls below 2x):

1. Each sample maps to its ADC level ``round((x + 32768) * (2**adc_bits - 1) / 65535)``; the
   exact 16-bit value of a level is sent once, as a deviation from the level's nominal value.
   ``adc_bits=16`` makes the mapping the identity, so any int16 stream is accepted.
2. Levels are numbered densely: each group counts its levels, and a level far rarer than both
   neighbours (an ADC missing code) leaves the numbering, so a signal crossing it moves by one
   rank instead of two. Both sides apply the same rule to the same counts; a sample that hits
   an excluded level is sent as a reserved escape plus the level in ``adc_bits`` plain bits.
3. The first difference of the level ranks is predicted by a sign-sign LMS filter of order 16
   (weights clamped to [-32768, 32767], history clamped to [-2048, 2047]; this bounds the
   arithmetic for vectorisation and affects only the prediction, never exactness).
4. The zigzagged prediction error is coded by an adaptive binary tree with an Elias-gamma
   escape, through an LZMA-style binary range coder with 12-bit probabilities.

Channels split into contiguous groups, each with its own range coder and level table; groups
code in parallel in the Rust engine. Within a group each channel's samples of the window are
coded together (channel-major), which keeps one channel's state in cache. The production implementation is the Rust one in
``sc_neurocore_engine`` (``backend="rust"`` or ``"auto"``); this module's pure-Python coder is the
reference it is tested against (identical bytes) and is far too slow for real-time use.
"""

from __future__ import annotations

import importlib
import operator
from types import ModuleType
from typing import Any

import numpy as np

PROB_BITS = 12
ADAPT_SHIFT = 6
LMS_ORDER = 16
LMS_SHIFT = 12
LMS_STEP = 2
WEIGHT_MIN, WEIGHT_MAX = -32768, 32767
HISTORY_MIN, HISTORY_MAX = -2048, 2047
TREE_ESCAPE = 63
MAX_ESCAPE_BITS = 48
EXCLUDE_RATIO = 64
EXCLUDE_MIN_COUNT = 8
INCLUDE_RATIO = 16
COUNT_LIMIT = 1 << 16
STREAM_BACKENDS = ("auto", "python", "rust")
_PROB_INIT = 1 << (PROB_BITS - 1)
_TOP = 1 << 24
_MASK32 = 0xFFFFFFFF


def level_of(sample: int, bits: int) -> int:
    """Return the ADC level of an int16 sample for a ``bits``-bit ADC scaled to int16."""
    numerator = (sample + 32768) * ((1 << bits) - 1)
    return (2 * numerator + 65535) // 131070


def nominal_value(level: int, bits: int) -> int:
    """Return the nominal int16 value of an ADC level (inverse scaling, rounded half up)."""
    scale = (1 << bits) - 1
    return (2 * level * 65535 + scale) // (2 * scale) - 32768


def detect_adc_bits(samples: np.ndarray[Any, Any], backend: str = "auto") -> int:
    """Return the smallest ADC width in 1..16 whose level mapping is one-to-one on ``samples``.

    Parameters
    ----------
    samples : ndarray
        Integer samples in the int16 range (any shape).
    backend : str
        ``"auto"``, ``"rust"`` or ``"python"``.

    Returns
    -------
    int
        The ADC width; 16 always qualifies.

    Raises
    ------
    ValueError
        For non-integer samples or samples outside the int16 range.
    """
    array = np.asarray(samples)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        if not bool(np.all(np.isfinite(array))) or not bool(np.all(array == np.round(array))):
            raise ValueError("ADC detection needs integer samples")
    if array.size and (array.min() < -32768 or array.max() > 32767):
        raise ValueError("ADC detection needs samples in the int16 range")
    flat = np.ascontiguousarray(array.astype(np.int16).reshape(-1))
    native = _native(backend)
    if native is not None:
        return int(native.waveform_codec_detect_adc_bits(flat))
    values = np.unique(flat).astype(np.int64)
    for bits in range(1, 16):
        levels = ((values + 32768) * ((1 << bits) - 1) * 2 + 65535) // 131070
        if len(np.unique(levels)) == len(values):
            return bits
    return 16


def _native(backend: str) -> ModuleType | None:
    """Return the engine module with the stream coder, or None for the Python reference."""
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
    if engine is not None and hasattr(engine, "WaveformLosslessEncoder"):
        return engine
    if backend == "rust":
        raise ValueError(
            "backend 'rust' needs sc_neurocore_engine with the streaming lossless coder"
        )
    return None


def _sample_count(samples: Any) -> int:
    """Return ``samples`` as an int in ``[0, 2**64)``; refuse anything else before any decoding."""
    try:
        count = operator.index(samples)
    except TypeError:
        raise ValueError(f"samples must be an integer, got {samples!r}") from None
    if not 0 <= count < 1 << 64:
        raise ValueError(f"samples must be in [0, 2**64), got {count}")
    return count


def _validate(channels: int, adc_bits: int, groups: int) -> None:
    if channels < 1:
        raise ValueError("a lossless stream needs at least one channel")
    if not 1 <= adc_bits <= 16:
        raise ValueError("adc_bits must be 1..16")
    if not 1 <= groups <= min(channels, 65535):
        raise ValueError("groups must be 1..min(channels, 65535)")


class _RangeEncoder:
    """LZMA-style binary range encoder (Python reference)."""

    def __init__(self) -> None:
        self.low = 0
        self.range = _MASK32
        self.cache = 0
        self.cache_size = 1
        self.out = bytearray()

    def _shift_low(self) -> None:
        if (self.low & _MASK32) < 0xFF000000 or (self.low >> 32) != 0:
            carry = self.low >> 32
            temp = self.cache
            while True:
                self.out.append((temp + carry) & 0xFF)
                temp = 0xFF
                self.cache_size -= 1
                if self.cache_size == 0:
                    break
            self.cache = (self.low & _MASK32) >> 24
        self.cache_size += 1
        self.low = ((self.low & _MASK32) << 8) & _MASK32

    def _normalize(self) -> None:
        while self.range < _TOP:
            self.range = (self.range << 8) & _MASK32
            self._shift_low()

    def bit(self, probs: list[int], index: int, bit: int) -> None:
        prob = probs[index]
        bound = (self.range >> PROB_BITS) * prob
        if bit:
            self.low += bound
            self.range -= bound
            probs[index] = prob - (prob >> ADAPT_SHIFT)
        else:
            self.range = bound
            probs[index] = prob + (((1 << PROB_BITS) - prob) >> ADAPT_SHIFT)
        self._normalize()

    def bit_precise(self, probs: list[int], index: int, bit: int, shift: int) -> None:
        """Code a bit with a 16-bit probability adapted by ``shift`` (rare events)."""
        prob = probs[index]
        bound = (self.range >> 16) * prob
        if bit:
            self.low += bound
            self.range -= bound
            probs[index] = prob - (prob >> shift)
        else:
            self.range = bound
            probs[index] = prob + ((65536 - prob) >> shift)
        self._normalize()

    def direct(self, value: int, count: int) -> None:
        for i in range(count - 1, -1, -1):
            self.range >>= 1
            if (value >> i) & 1:
                self.low += self.range
            self._normalize()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class _RangeDecoder:
    """LZMA-style binary range decoder (Python reference).

    ``data`` has passed the packet framing checks (at least 5 bytes, first byte zero).
    """

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.range = _MASK32
        self.code = 0
        for _ in range(5):
            self.code = ((self.code << 8) | self._next()) & _MASK32

    def _next(self) -> int:
        if self.pos >= len(self.data):
            raise ValueError("corrupted lossless packet: range coder ran past its bytes")
        byte = self.data[self.pos]
        self.pos += 1
        return byte

    def _normalize(self) -> None:
        while self.range < _TOP:
            self.range = (self.range << 8) & _MASK32
            self.code = ((self.code << 8) | self._next()) & _MASK32

    def bit(self, probs: list[int], index: int) -> int:
        prob = probs[index]
        bound = (self.range >> PROB_BITS) * prob
        if self.code < bound:
            self.range = bound
            probs[index] = prob + (((1 << PROB_BITS) - prob) >> ADAPT_SHIFT)
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            probs[index] = prob - (prob >> ADAPT_SHIFT)
            bit = 1
        self._normalize()
        return bit

    def bit_precise(self, probs: list[int], index: int, shift: int) -> int:
        """Decode a bit coded by ``_RangeEncoder.bit_precise``."""
        prob = probs[index]
        bound = (self.range >> 16) * prob
        if self.code < bound:
            self.range = bound
            probs[index] = prob + ((65536 - prob) >> shift)
            bit = 0
        else:
            self.code -= bound
            self.range -= bound
            probs[index] = prob - (prob >> shift)
            bit = 1
        self._normalize()
        return bit

    def direct(self, count: int) -> int:
        value = 0
        for _ in range(count):
            self.range >>= 1
            bit = 1 if self.code >= self.range else 0
            if bit:
                self.code -= self.range
            value = (value << 1) | bit
            self._normalize()
        return value

    def finish(self) -> None:
        if self.pos != len(self.data):
            raise ValueError("corrupted lossless packet: trailing range-coder bytes")


class _ResidualModel:
    """Adaptive 6-bit tree for values below 63 and an adaptive Elias-gamma escape."""

    def __init__(self) -> None:
        self.tree = [_PROB_INIT] * 64
        self.escape = [_PROB_INIT] * MAX_ESCAPE_BITS

    def put(self, encoder: _RangeEncoder, u: int) -> None:
        """Code ``u``; values of 63 and more escape with the gamma code of ``u - 61``."""
        symbol = min(u, TREE_ESCAPE)
        self._put_symbol(encoder, symbol)
        if symbol == TREE_ESCAPE:
            self._put_gamma(encoder, u - TREE_ESCAPE + 2)

    def put_marker(self, encoder: _RangeEncoder) -> None:
        """Code the reserved escape value 1 (the next item is not a residual)."""
        self._put_symbol(encoder, TREE_ESCAPE)
        self._put_gamma(encoder, 1)

    def _put_symbol(self, encoder: _RangeEncoder, symbol: int) -> None:
        node = 1
        for i in range(5, -1, -1):
            bit = (symbol >> i) & 1
            encoder.bit(self.tree, node, bit)
            node = node * 2 + bit

    def _put_gamma(self, encoder: _RangeEncoder, value: int) -> None:
        length = value.bit_length()
        for i in range(length - 1):
            encoder.bit(self.escape, i, 1)
        encoder.bit(self.escape, length - 1, 0)
        encoder.direct(value, length - 1)

    def get(self, decoder: _RangeDecoder) -> int | None:
        """Decode a residual, or None for the reserved marker."""
        node = 1
        for _ in range(6):
            node = node * 2 + decoder.bit(self.tree, node)
        symbol = node - 64
        if symbol < TREE_ESCAPE:
            return symbol
        length = 1
        while decoder.bit(self.escape, length - 1):
            length += 1
            if length > MAX_ESCAPE_BITS:
                raise ValueError("corrupted lossless packet: escape too long")
        value = (1 << (length - 1)) | decoder.direct(length - 1)
        return value + TREE_ESCAPE - 2 if value > 1 else None

    def get_residual(self, decoder: _RangeDecoder) -> int:
        """Decode a residual; the marker is refused (the deviation model never sends it)."""
        value = self.get(decoder)
        if value is None:
            raise ValueError("corrupted lossless packet: unexpected level marker")
        return value


def _zigzag(value: int) -> int:
    return value << 1 if value >= 0 else ((-value) << 1) - 1


def _unzigzag(value: int) -> int:
    return value >> 1 if value & 1 == 0 else -((value >> 1) + 1)


class _Channel:
    """Per-channel predictor state and residual model."""

    def __init__(self, bits: int) -> None:
        self.previous = 1 << (bits - 1)
        self.history = [0] * LMS_ORDER
        self.weights = [0] * LMS_ORDER
        self.model = _ResidualModel()

    def predict(self) -> int:
        return sum(w * h for w, h in zip(self.weights, self.history, strict=True)) >> LMS_SHIFT

    def update(self, difference: int, error: int) -> None:
        if error:
            step = LMS_STEP if error > 0 else -LMS_STEP
            for i, h in enumerate(self.history):
                if h:
                    weight = self.weights[i] + (step if h > 0 else -step)
                    self.weights[i] = min(max(weight, WEIGHT_MIN), WEIGHT_MAX)
        clamped = min(max(difference, HISTORY_MIN), HISTORY_MAX)
        self.history = [clamped, *self.history[:-1]]


class _LevelRanks:
    """Dense numbering of a group's levels with adaptive exclusion of missing codes."""

    def __init__(self, bits: int) -> None:
        levels = 1 << bits
        self.count = [0] * levels
        self.excluded = [False] * levels
        self.rank = list(range(levels))
        self.level_at = list(range(levels))

    def _exclusion(self, level: int, now: bool) -> bool:
        if level == 0 or level + 1 >= len(self.count):
            return False
        rarer = min(self.count[level - 1], self.count[level + 1])
        own = self.count[level]
        if now:
            return own * INCLUDE_RATIO < rarer
        return rarer >= EXCLUDE_MIN_COUNT and own * EXCLUDE_RATIO < rarer

    def observe(self, level: int) -> None:
        """Count one occurrence of ``level`` and update the numbering."""
        self.count[level] += 1
        if self.count[level] >= COUNT_LIMIT:
            self.count = [count >> 1 for count in self.count]
            span = range(len(self.count))
        else:
            span = range(max(level - 1, 0), min(level + 2, len(self.count)))
        changed = False
        for candidate in span:
            now = self.excluded[candidate]
            nxt = self._exclusion(candidate, now)
            if nxt != now:
                self.excluded[candidate] = nxt
                changed = True
        if changed:
            self.level_at = []
            for level_index, excluded in enumerate(self.excluded):
                self.rank[level_index] = len(self.level_at)
                if not excluded:
                    self.level_at.append(level_index)


class _Group:
    """Channels of one group, its level table, numbering and deviation model."""

    def __init__(self, start: int, end: int, bits: int) -> None:
        self.start = start
        self.end = end
        self.channels = [_Channel(bits) for _ in range(end - start)]
        self.table: dict[int, int] = {}
        self.deviation = _ResidualModel()
        self.ranks = _LevelRanks(bits)


def _groups(channels: int, groups: int, bits: int) -> list[_Group]:
    return [
        _Group(g * channels // groups, (g + 1) * channels // groups, bits) for g in range(groups)
    ]


class _PyEncoder:
    """Pure-Python reference encoder."""

    def __init__(self, channels: int, adc_bits: int, groups: int) -> None:
        self.channels, self.bits = channels, adc_bits
        self.groups = _groups(channels, groups, adc_bits)

    def encode(self, window: np.ndarray[Any, Any]) -> bytes:
        bits = self.bits
        columns = window.astype(np.int64).T.tolist()
        for group in self.groups:
            pending: dict[int, int] = {}
            for index in range(group.start, group.end):
                for value in columns[index]:
                    level = level_of(value, bits)
                    known = group.table.get(level, pending.get(level))
                    if known is None:
                        pending[level] = value
                    elif known != value:
                        raise ValueError(
                            f"sample {value} in channel {index} contradicts {known} "
                            f"at {bits}-bit ADC level {level}"
                        )
        packet = bytearray()
        for group in self.groups:
            encoder = _RangeEncoder()
            ranks = group.ranks
            for offset, channel in enumerate(group.channels):
                for value in columns[group.start + offset]:
                    level = level_of(value, bits)
                    difference = ranks.rank[level] - ranks.rank[channel.previous]
                    error = difference - channel.predict()
                    if ranks.excluded[level]:
                        channel.model.put_marker(encoder)
                        encoder.direct(level, bits)
                    else:
                        channel.model.put(encoder, _zigzag(error))
                    channel.update(difference, error)
                    channel.previous = level
                    if level not in group.table:
                        group.deviation.put(encoder, _zigzag(value - nominal_value(level, bits)))
                        group.table[level] = value
                    ranks.observe(level)
            body = encoder.finish()
            packet += len(body).to_bytes(4, "big") + body
        return bytes(packet)


class _PyDecoder:
    """Pure-Python reference decoder."""

    def __init__(self, channels: int, adc_bits: int, groups: int) -> None:
        self.channels, self.bits = channels, adc_bits
        self.groups = _groups(channels, groups, adc_bits)
        self.broken = False

    def decode(self, packet: bytes, samples: int) -> np.ndarray[Any, Any]:
        if self.broken:
            raise ValueError(
                "lossless stream decoder is broken by an earlier corrupted packet; start a new decoder"
            )
        bodies = []
        pos = 0
        for _ in self.groups:
            if pos + 4 > len(packet):
                raise ValueError("corrupted lossless packet: missing group length")
            length = int.from_bytes(packet[pos : pos + 4], "big")
            if pos + 4 + length > len(packet):
                raise ValueError("corrupted lossless packet: group longer than packet")
            body = packet[pos + 4 : pos + 4 + length]
            if len(body) < 5 or body[0] != 0:
                raise ValueError("corrupted lossless packet: bad range-coder start")
            bodies.append(body)
            pos += 4 + length
        if pos != len(packet):
            raise ValueError("corrupted lossless packet: trailing bytes")
        for group, body in zip(self.groups, bodies, strict=True):
            if samples * (group.end - group.start) > 64 * (len(body) + 8):
                raise ValueError(
                    "corrupted lossless packet: more samples requested than the packet can hold"
                )
        try:
            return self._decode_groups(bodies, samples)
        except ValueError:
            self.broken = True
            raise

    def _decode_groups(self, bodies: list[bytes], samples: int) -> np.ndarray[Any, Any]:
        bits = self.bits
        window = np.zeros((samples, self.channels), dtype=np.int16)
        for group, body in zip(self.groups, bodies, strict=True):
            decoder = _RangeDecoder(body)
            ranks = group.ranks
            for offset, channel in enumerate(group.channels):
                for row in range(samples):
                    previous_rank = ranks.rank[channel.previous]
                    prediction = channel.predict()
                    u = channel.model.get(decoder)
                    if u is not None:
                        error = _unzigzag(u)
                        difference = prediction + error
                        rank = previous_rank + difference
                        if not 0 <= rank < len(ranks.level_at):
                            raise ValueError("corrupted lossless packet: ADC level out of range")
                        level = ranks.level_at[rank]
                    else:
                        level = decoder.direct(bits)
                        if not ranks.excluded[level]:
                            raise ValueError(
                                "corrupted lossless packet: escaped level is not excluded"
                            )
                        difference = ranks.rank[level] - previous_rank
                        error = difference - prediction
                    channel.update(difference, error)
                    channel.previous = level
                    value = group.table.get(level)
                    if value is None:
                        value = nominal_value(level, bits) + _unzigzag(
                            group.deviation.get_residual(decoder)
                        )
                        if not -32768 <= value <= 32767:
                            raise ValueError("corrupted lossless packet: sample outside int16")
                        group.table[level] = value
                    ranks.observe(level)
                    window[row, group.start + offset] = value
            decoder.finish()
        return window


def _window(window: np.ndarray[Any, Any], channels: int) -> np.ndarray[Any, Any]:
    array = np.asarray(window)
    if array.ndim != 2 or array.shape[1] != channels:
        raise ValueError(f"window must be (samples, {channels})")
    if array.size and not np.issubdtype(array.dtype, np.integer):
        if not bool(np.all(np.isfinite(array))) or not bool(np.all(array == np.round(array))):
            raise ValueError("lossless coding needs integer samples")
    if array.size and (array.min() < -32768 or array.max() > 32767):
        raise ValueError("lossless coding needs samples in the int16 range")
    return np.ascontiguousarray(array, dtype=np.int16)


class LosslessStreamEncoder:
    """Stateful lossless encoder for a multi-channel int16 stream.

    Parameters
    ----------
    channels : int
        Channels per window.
    adc_bits : int
        ADC width 1..16 (16 accepts any int16 stream; see ``detect_adc_bits``).
    groups : int
        Contiguous channel groups, each with its own range coder (coded in parallel in Rust).
    backend : str
        ``"auto"`` (Rust when installed), ``"rust"`` or ``"python"``; identical packets.
    """

    def __init__(
        self, channels: int, adc_bits: int = 16, groups: int = 1, backend: str = "auto"
    ) -> None:
        _validate(channels, adc_bits, groups)
        self.channels = channels
        native = _native(backend)
        self._coder: Any = (
            native.WaveformLosslessEncoder(channels, adc_bits, groups)
            if native is not None
            else _PyEncoder(channels, adc_bits, groups)
        )

    def encode(self, window: np.ndarray[Any, Any]) -> bytes:
        """Encode a ``(samples, channels)`` window of int16-range integers into a packet.

        Raises
        ------
        ValueError
            For a malformed window or a sample that contradicts the stream's ADC levels
            (the encoder state is then unchanged).
        """
        return bytes(self._coder.encode(_window(window, self.channels)))


class LosslessStreamDecoder:
    """Stateful lossless decoder matching ``LosslessStreamEncoder`` (same parameters)."""

    def __init__(
        self, channels: int, adc_bits: int = 16, groups: int = 1, backend: str = "auto"
    ) -> None:
        _validate(channels, adc_bits, groups)
        self.channels = channels
        native = _native(backend)
        self._coder: Any = (
            native.WaveformLosslessDecoder(channels, adc_bits, groups)
            if native is not None
            else _PyDecoder(channels, adc_bits, groups)
        )

    def decode(self, packet: bytes, samples: int) -> np.ndarray[Any, Any]:
        """Decode a packet of ``samples`` rows into an int16 ``(samples, channels)`` window.

        Raises
        ------
        ValueError
            For a malformed packet. A framing error (group lengths, trailing bytes, range-coder
            start) leaves the decoder unchanged; content found invalid while decoding breaks the
            stream state, and the decoder then refuses every later packet (start a new one). A
            sample count that is not an integer in ``[0, 2**64)`` is refused before decoding.
        """
        count = _sample_count(samples)
        return np.asarray(self._coder.decode(bytes(packet), count), dtype=np.int16)
