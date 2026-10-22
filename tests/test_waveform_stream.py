# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Streaming lossless coder: Rust packets equal the Python reference

"""Round trips, Rust/Python packet identity, ADC detection and refusals of the streaming coder."""

from __future__ import annotations

import struct
import sys
from typing import Any

import numpy as np
import pytest

from sc_neurocore.spike_codec import waveform_stream as ws
from sc_neurocore.spike_codec.waveform_codec import WaveformCodec
from sc_neurocore.spike_codec.waveform_stream import (
    LosslessStreamDecoder,
    LosslessStreamEncoder,
    detect_adc_bits,
)

BACKENDS = ("python", "rust")


def _adc_stream(samples: int, channels: int, seed: int) -> np.ndarray[Any, Any]:
    """Return a 10-bit ADC style int16 stream (oscillation plus noise, scaled by 65535/1023)."""
    rng = np.random.default_rng(seed)
    t = np.arange(samples)[:, None]
    levels = (
        512
        + 300 * np.sin(t / (40.0 + np.arange(channels)))
        + rng.integers(-40, 41, (samples, channels))
    )
    levels = np.clip(np.round(levels), 0, 1023).astype(np.int64)
    return np.vectorize(lambda level: ws.nominal_value(int(level), 10))(levels).astype(np.int16)


def _packets(
    data: np.ndarray[Any, Any], bits: int, groups: int, window: int, backend: str
) -> list[bytes]:
    encoder = LosslessStreamEncoder(data.shape[1], bits, groups, backend)
    return [encoder.encode(data[i : i + window]) for i in range(0, len(data), window)]


@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize("window", [1, 17, 300])
def test_rust_packets_equal_the_python_reference(groups: int, window: int) -> None:
    """Every packet is byte-identical across backends, and both decoders return the input."""
    data = _adc_stream(300, 6, 1)
    packets = {backend: _packets(data, 10, groups, window, backend) for backend in BACKENDS}
    assert packets["python"] == packets["rust"]
    for backend in BACKENDS:
        decoder = LosslessStreamDecoder(6, 10, groups, backend)
        rows = [
            decoder.decode(p, min(window, 300 - i * window))
            for i, p in enumerate(packets["python"])
        ]
        np.testing.assert_array_equal(np.concatenate(rows), data)


def test_full_range_noise_escapes_and_carries_match() -> None:
    """Full-range 16-bit noise exercises escapes, direct bits and carry propagation identically."""
    rng = np.random.default_rng(2)
    noise = rng.integers(-32768, 32768, size=(400, 2)).astype(np.int16)
    noise[:6, 0] = [32767, -32768, 32767, -32768, 0, 32767]
    packets = {backend: _packets(noise, 16, 1, 400, backend) for backend in BACKENDS}
    assert packets["python"] == packets["rust"]
    for backend in BACKENDS:
        decoded = LosslessStreamDecoder(2, 16, 1, backend).decode(packets["python"][0], 400)
        np.testing.assert_array_equal(decoded, noise)


def _missing_code_stream(samples: int, channels: int, rare: int) -> np.ndarray[Any, Any]:
    """Return narrow noise around ADC level 500, a missing code kept only every ``rare``-th sample."""
    rng = np.random.default_rng(9)
    t = np.arange(samples)[:, None]
    levels = 500 + np.round(2 * np.sin(t / 50.0)) + rng.integers(-3, 4, (samples, channels))
    levels = levels.astype(np.int64)
    index = np.arange(levels.size).reshape(levels.shape)
    levels[(levels == 500) & (index % rare != 0)] = 501
    return np.vectorize(lambda level: ws.nominal_value(int(level), 10))(levels).astype(np.int16)


def test_missing_codes_leave_the_numbering_identically() -> None:
    """A missing code costs almost nothing; rare hits on it decode, with equal packets."""
    gap = _missing_code_stream(3000, 3, 10**9)
    levels = (gap.astype(np.int64) + 32768) * 1023 * 2 + 65535
    levels //= 131070
    dense = np.vectorize(lambda level: ws.nominal_value(int(level) - (level > 500), 10))(levels)
    sizes = [
        len(LosslessStreamEncoder(3, 10, 1, "rust").encode(data))
        for data in (gap, dense.astype(np.int16))
    ]
    assert sizes[0] < 1.01 * sizes[1]
    hits = _missing_code_stream(3000, 3, 97)
    for groups, window in [(1, 3000), (3, 20)]:
        packets = {backend: _packets(hits, 10, groups, window, backend) for backend in BACKENDS}
        assert packets["python"] == packets["rust"]
        for backend in BACKENDS:
            decoder = LosslessStreamDecoder(3, 10, groups, backend)
            rows = [decoder.decode(p, window) for p in packets["python"]]
            np.testing.assert_array_equal(np.concatenate(rows), hits)


def test_level_counts_halve_identically() -> None:
    """A level passing 65536 occurrences halves all counts; both backends agree."""
    rng = np.random.default_rng(10)
    draws = rng.integers(0, 20, 80000)
    levels = np.select([draws == 0, draws == 1, draws == 2], [511, 513, 514], 512)
    data = np.vectorize(lambda level: ws.nominal_value(int(level), 10))(levels)
    data = data.astype(np.int16).reshape(-1, 1)
    packets = {backend: _packets(data, 10, 1, 40000, backend) for backend in BACKENDS}
    assert packets["python"] == packets["rust"]
    decoder = LosslessStreamDecoder(1, 10, 1, "rust")
    rows = [decoder.decode(p, 40000) for p in packets["python"]]
    np.testing.assert_array_equal(np.concatenate(rows), data)


@pytest.mark.parametrize("backend", BACKENDS)
def test_misplaced_level_markers_are_refused(backend: str) -> None:
    """A marker before a level that is not excluded, or in place of a deviation, is refused."""

    def packet(write: Any) -> bytes:
        encoder = ws._RangeEncoder()
        write(encoder, ws._ResidualModel(), ws._ResidualModel())
        body = encoder.finish()
        return len(body).to_bytes(4, "big") + body

    def marker_then_level(encoder: Any, residuals: Any, deviations: Any) -> None:
        residuals.put_marker(encoder)
        encoder.direct(512, 10)

    def marker_as_deviation(encoder: Any, residuals: Any, deviations: Any) -> None:
        residuals.put(encoder, 0)
        deviations.put_marker(encoder)

    def escape_too_long(encoder: Any, residuals: Any, deviations: Any) -> None:
        for node in (1, 3, 7, 15, 31, 63):  # the tree path of the escape symbol 63
            encoder.bit(residuals.tree, node, 1)
        for index in range(ws.MAX_ESCAPE_BITS):
            encoder.bit(residuals.escape, index, 1)

    def rank_below_zero(encoder: Any, residuals: Any, deviations: Any) -> None:
        residuals.put(encoder, ws._zigzag(-600))

    def value_outside_int16(encoder: Any, residuals: Any, deviations: Any) -> None:
        residuals.put(encoder, 0)
        deviations.put(encoder, ws._zigzag(40000))

    for write, message in [
        (marker_then_level, "escaped level is not excluded"),
        (marker_as_deviation, "unexpected level marker"),
        (escape_too_long, "escape too long"),
        (rank_below_zero, "ADC level out of range"),
        (value_outside_int16, "sample outside int16"),
    ]:
        with pytest.raises(ValueError, match=message):
            LosslessStreamDecoder(1, 10, 1, backend).decode(packet(write), 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_range_coder_byte_count_is_checked(backend: str) -> None:
    """A body too short for the requested samples, or with unused bytes, is refused."""
    data = _adc_stream(100, 1, 11)
    body = LosslessStreamEncoder(1, 10, 1, backend).encode(data)[4:]
    for broken, samples, message in [
        (b"\x00" * 5, 100, "ran past its bytes"),
        (body + b"\x00" * 8, 100, "trailing range-coder bytes"),
    ]:
        packet = len(broken).to_bytes(4, "big") + broken
        with pytest.raises(ValueError, match=message):
            LosslessStreamDecoder(1, 10, 1, backend).decode(packet, samples)


@pytest.mark.parametrize("backend", BACKENDS)
def test_oversized_sample_requests_are_refused_before_allocation(backend: str) -> None:
    """A request the packet cannot hold (e.g. 2**63 samples) is refused; the decoder stays usable."""
    packet = LosslessStreamEncoder(2, 1, 1, backend).encode(np.zeros((1, 2), dtype=np.int16))
    decoder = LosslessStreamDecoder(2, 1, 1, backend)
    for samples in (1 << 63, 1000):
        with pytest.raises(ValueError, match="more samples requested than the packet can hold"):
            decoder.decode(packet, samples)
    np.testing.assert_array_equal(decoder.decode(packet, 1), np.zeros((1, 2), dtype=np.int16))


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_sample_counts_are_refused_before_decoding(backend: str) -> None:
    """Negative, fractional, non-numeric and too large counts are refused; the decoder stays usable."""
    data = _adc_stream(40, 2, 12)
    encoder = LosslessStreamEncoder(2, 10, 1, backend)
    decoder = LosslessStreamDecoder(2, 10, 1, backend)
    first, second = encoder.encode(data[:20]), encoder.encode(data[20:])
    np.testing.assert_array_equal(decoder.decode(first, np.int64(20)), data[:20])
    for samples in (-1, 1.5, "20", 1 << 64):
        with pytest.raises(ValueError, match="samples must be"):
            decoder.decode(second, samples)
    np.testing.assert_array_equal(decoder.decode(second, 20), data[20:])


def test_adc_detection_refuses_values_it_would_change() -> None:
    """Fractional or out-of-range samples are refused instead of being cast to int16."""
    for backend in BACKENDS:
        for values, message in [([0.1, 0.2], "integer samples"), ([0, 65536], "int16 range")]:
            with pytest.raises(ValueError, match=message):
                detect_adc_bits(np.array(values), backend)
        assert detect_adc_bits(np.array([64.0, -64.0]), backend) == 1, (
            "integral floats are accepted"
        )


def test_adc_detection_agrees() -> None:
    """Both backends find 10 bits for an ADC stream and 16 for full-range noise."""
    rng = np.random.default_rng(3)
    noise = rng.integers(-32768, 32768, size=5000).astype(np.int16)
    for backend in BACKENDS:
        assert detect_adc_bits(_adc_stream(400, 3, 4), backend) == 10
        assert detect_adc_bits(noise, backend) == 16
        assert detect_adc_bits(np.zeros(3, dtype=np.int16), backend) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_contradicting_sample_is_refused_and_state_kept(backend: str) -> None:
    """A sample that breaks the ADC level table is refused; the encoder continues unchanged."""
    data = _adc_stream(60, 2, 5)
    encoder = LosslessStreamEncoder(2, 10, 1, backend)
    reference = LosslessStreamEncoder(2, 10, 1, backend)
    encoder.encode(data[:40])
    reference.encode(data[:40])
    bad = data[40:].copy()
    bad[0, 0] = data[0, 0] + 1
    with pytest.raises(ValueError, match="contradicts"):
        encoder.encode(np.concatenate([data[:1], bad]))
    assert encoder.encode(data[40:]) == reference.encode(data[40:])


@pytest.mark.parametrize("backend", BACKENDS)
def test_corrupted_packets_are_refused(backend: str) -> None:
    """Truncated, padded and broken packets raise ValueError; the decoder state is kept."""
    data = _adc_stream(200, 4, 6)
    packet = LosslessStreamEncoder(4, 10, 2, backend).encode(data)

    def fresh() -> LosslessStreamDecoder:
        return LosslessStreamDecoder(4, 10, 2, backend)

    for broken, message in [
        (packet[:3], "missing group length"),
        (packet[:-1], "longer than packet"),
        (packet + b"\x00", "trailing bytes"),
        (packet[:4] + b"\x01" + packet[5:], "range-coder start"),
    ]:
        with pytest.raises(ValueError, match=message):
            fresh().decode(broken, 200)
    broken = fresh()
    with pytest.raises(ValueError, match="corrupted lossless packet"):
        broken.decode(packet, 201)
    with pytest.raises(ValueError, match="start a new decoder"):
        broken.decode(packet, 200)
    decoder = fresh()
    with pytest.raises(ValueError):
        decoder.decode(packet[:3], 200)
    np.testing.assert_array_equal(decoder.decode(packet, 200), data)


@pytest.mark.parametrize("backend", BACKENDS)
def test_configuration_and_window_refusals(backend: str) -> None:
    """Invalid channel, width and group settings and malformed windows are refused."""
    for channels, bits, groups in [(0, 10, 1), (4, 0, 1), (4, 17, 1), (4, 10, 5), (4, 10, 0)]:
        with pytest.raises(ValueError):
            LosslessStreamEncoder(channels, bits, groups, backend)
        with pytest.raises(ValueError):
            LosslessStreamDecoder(channels, bits, groups, backend)
    encoder = LosslessStreamEncoder(2, 16, 1, backend)
    with pytest.raises(ValueError, match=r"\(samples, 2\)"):
        encoder.encode(np.zeros((4, 3), dtype=np.int16))
    with pytest.raises(ValueError, match="integer samples"):
        encoder.encode(np.full((2, 2), 0.5))
    with pytest.raises(ValueError, match="int16 range"):
        encoder.encode(np.full((2, 2), 40000))
    integral = np.full((2, 2), 3.0)
    decoded = LosslessStreamDecoder(2, 16, 1, backend).decode(encoder.encode(integral), 2)
    np.testing.assert_array_equal(decoded, integral)
    empty = encoder.encode(np.zeros((0, 2), dtype=np.int16))
    assert LosslessStreamDecoder(2, 16, 1, backend).decode(empty, 0).shape == (0, 2)


def test_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown backends are refused; without the engine auto falls back and rust refuses."""
    with pytest.raises(ValueError, match="backend must be one of"):
        LosslessStreamEncoder(1, 16, 1, "gpu")
    monkeypatch.setitem(sys.modules, "sc_neurocore_engine.sc_neurocore_engine", None)
    assert isinstance(LosslessStreamEncoder(1, 16, 1, "auto")._coder, ws._PyEncoder)
    with pytest.raises(ValueError, match="backend 'rust' needs"):
        LosslessStreamDecoder(1, 16, 1, "rust")


@pytest.mark.parametrize("backend", BACKENDS)
def test_waveform_codec_lossless_uses_the_stream_coder(backend: str) -> None:
    """``mode='lossless'`` writes mode byte 4, detects the ADC width and round-trips exactly."""
    data = _adc_stream(500, 3, 7)
    stream, result = WaveformCodec(mode="lossless", backend=backend).compress(data)
    assert stream[19] == 4 and result.lossless_waveform and result.compressed_bytes == len(stream)
    body_at = 4 + struct.calcsize("!IIHHBBBB") + 4
    assert stream[body_at] == 10, "detected ADC width"
    decoded = WaveformCodec.decompress(stream, backend=backend)
    assert decoded.mode == "lossless" and decoded.lossless_waveform
    np.testing.assert_array_equal(decoded.waveform, data)
    forced, _ = WaveformCodec(mode="lossless", adc_bits=16, backend=backend).compress(data)
    assert forced[body_at] == 16
    np.testing.assert_array_equal(WaveformCodec.decompress(forced, backend=backend).waveform, data)


def test_waveform_codec_lossless_refusals() -> None:
    """A bad ADC width argument or stream byte is refused; the Rice mode still round-trips."""
    with pytest.raises(ValueError, match="adc_bits"):
        WaveformCodec(mode="lossless", adc_bits=17)
    data = _adc_stream(100, 2, 8)
    stream, _ = WaveformCodec(mode="lossless").compress(data)
    body_at = 4 + struct.calcsize("!IIHHBBBB") + 4
    broken = stream[:body_at] + b"\x00" + stream[body_at + 1 : -4]
    from sc_neurocore.spike_codec import waveform_codec as wc

    with pytest.raises(ValueError, match="invalid ADC width"):
        WaveformCodec.decompress(wc._seal(broken))
    rice, _ = WaveformCodec(mode="lossless_rice").compress(data)
    assert rice[19] == 3
    np.testing.assert_array_equal(WaveformCodec.decompress(rice).waveform, data)
