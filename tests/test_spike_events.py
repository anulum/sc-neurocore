# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-event layer: Rust packets equal the Python reference

"""Spike detection, Rust/Python packet identity, window independence and refusals of the spike-event layer."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from sc_neurocore.spike_codec import spike_events as se
from sc_neurocore.spike_codec import waveform_stream as ws
from sc_neurocore.spike_codec.spike_events import (
    SpikeEventDecoder,
    SpikeEventEncoder,
    SpikeEventWindow,
    highpass_coefficients,
    lowpass_coefficients,
    threshold_factor,
)

BACKENDS = ("python", "rust")
RATE = 19531.0
SHAPE = np.array([-3000, -9000, -12000, -7000, -2000, 1500, 2500, 2000, 1000])


def _recording(
    samples: int, spikes: list[list[int]], seed: int = 1, shape: np.ndarray[Any, Any] = SHAPE
) -> np.ndarray[Any, Any]:
    """Return noise (about 1150 units RMS) with a slow drift and large spikes at the given samples."""
    rng = np.random.default_rng(seed)
    channels = len(spikes)
    t = np.arange(samples)[:, None]
    data = rng.integers(-1000, 1001, (4, samples, channels)).sum(axis=0)
    data = data + (3000 * np.sin(t / 3000.0)).astype(np.int64)
    for channel, times in enumerate(spikes):
        for start in times:
            data[start : start + len(shape), channel] += shape
    return np.clip(data, -32768, 32767).astype(np.int16)


def _run(
    data: np.ndarray[Any, Any], window: int, backend: str, **kwargs: Any
) -> tuple[list[bytes], list[SpikeEventWindow]]:
    encoder = SpikeEventEncoder(data.shape[1], RATE, backend=backend, **kwargs)
    s = encoder.settings
    decoder = SpikeEventDecoder(
        data.shape[1],
        s["bin_samples"],
        timing="sample" if s["sample_timing"] else "bin",
        power_bins=s["power_bins"],
        groups=s["groups"],
        backend=backend,
    )
    packets, windows = [], []
    for i in range(0, len(data), window):
        chunk = data[i : i + window]
        packets.append(encoder.encode(chunk))
        windows.append(decoder.decode(packets[-1], len(chunk)))
    return packets, windows


def _event_times(windows: list[SpikeEventWindow], bin_samples: int = 20) -> list[list[int]]:
    channels = windows[0].events.shape[1]
    times: list[list[int]] = [[] for _ in range(channels)]
    for w in windows:
        rows, columns = np.nonzero(w.events)
        for row, column in zip(rows, columns, strict=True):
            offset = 0 if w.offsets is None else int(w.offsets[row, column])
            times[column].append((w.first_bin + int(row)) * bin_samples + offset)
    return times


def test_filter_design_and_threshold_factor_agree_with_the_engine() -> None:
    """Q14 coefficients and the Q12 factor are identical in Rust and Python."""
    assert highpass_coefficients(250, RATE) == (15478, -30956, 15478, -30906, 14623)
    assert lowpass_coefficients(1000, RATE) == (343, 687, 343, -25407, 10397)
    assert threshold_factor(4.5) == 27327
    native = pytest.importorskip("sc_neurocore_engine.sc_neurocore_engine")
    for rate, high, low, sigma in [(RATE, 250.0, 1000.0, 4.5), (30000.0, 300.0, 3000.0, 3.5)]:
        rust = native.spike_events_filter_coefficients(rate, high, low, sigma)
        assert tuple(rust[0]) == highpass_coefficients(high, rate)
        assert tuple(rust[1]) == lowpass_coefficients(low, rate)
        assert rust[2] == threshold_factor(sigma)
    for cutoff in (10.0, 9000.0, float("nan")):
        with pytest.raises(ValueError, match="must lie in"):
            highpass_coefficients(cutoff, RATE)
    with pytest.raises(ValueError, match="cannot be represented"):
        lowpass_coefficients(20.0, RATE)


@pytest.mark.parametrize("timing", ["bin", "sample"])
@pytest.mark.parametrize(("window", "groups"), [(4000, 1), (20, 2), (7, 3)])
def test_rust_packets_equal_the_python_reference(timing: str, window: int, groups: int) -> None:
    """Every packet and every decoded window is identical across backends."""
    data = _recording(4000, [[2500, 3000], [], [2700]])
    runs = {
        b: _run(data, window, b, timing=timing, groups=groups, power_bins=5, warmup_samples=1000)
        for b in BACKENDS
    }
    assert runs["python"][0] == runs["rust"][0]
    for left, right in zip(runs["python"][1], runs["rust"][1], strict=True):
        assert left.first_bin == right.first_bin and left.first_period == right.first_period
        np.testing.assert_array_equal(left.events, right.events)
        np.testing.assert_array_equal(left.power, right.power)
        if timing == "sample":
            np.testing.assert_array_equal(left.offsets, right.offsets)
        else:
            assert left.offsets is None and right.offsets is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_large_spikes_are_found_once_each(backend: str) -> None:
    """Spikes after the warm-up are reported once, at their sample; warm-up and refractory hold."""
    spikes = [[500, 3000, 3010, 9000], [], [4000, 4030]]
    _, windows = _run(_recording(10000, spikes), 10000, backend, timing="sample", power_bins=0)
    times = _event_times(windows)
    assert [len(t) for t in times] == [2, 0, 2]
    for found, spike in zip(times[0] + times[2], [3000, 9000, 4000, 4030], strict=True):
        assert spike <= found <= spike + 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_refractory_period_ends_exactly_after_its_samples(backend: str) -> None:
    """Spikes 20 samples apart are both reported with a 20-sample refractory period; 19 apart, one."""
    sharp = np.array([-12000, -12000, -9000, -4000, 0, 2000, 2000, 1000])
    data = _recording(8000, [[5000, 5020], [5000, 5019]], shape=sharp)
    _, windows = _run(data, 8000, backend, timing="sample", power_bins=0)
    assert _event_times(windows) == [[5000, 5020], [5000]]


def test_windows_change_nothing_and_power_follows_the_noise() -> None:
    """Window lengths do not change events or band power; half the noise gives less power."""
    data = _recording(6000, [[2500], [3100], []])

    def flat(windows: list[SpikeEventWindow]) -> tuple[Any, ...]:
        return (
            np.concatenate([w.events for w in windows]),
            np.concatenate([w.power for w in windows]),
        )

    expected = flat(_run(data, 6000, "rust", power_bins=7)[1])
    assert expected[0].shape == (300, 3) and expected[1].shape == (42, 3)
    for window in (1, 13, 137):
        got = flat(_run(data, window, "rust", power_bins=7)[1])
        np.testing.assert_array_equal(got[0], expected[0])
        np.testing.assert_array_equal(got[1], expected[1])
    quiet = flat(_run(data // 2, 6000, "rust", power_bins=7)[1])
    assert quiet[1].sum() * 3 < expected[1].sum() * 2
    shifted = flat(_run(data, 6000, "rust", power_bins=7, power_shift=3)[1])
    assert shifted[1].max() <= expected[1].max() >> 3


def test_defaults_follow_the_sample_rate() -> None:
    """Without explicit values the bin is about 1 ms and the warm-up about 100 ms."""
    settings = SpikeEventEncoder(2, 30000.0, backend="python").settings
    assert settings["bin_samples"] == 30 and settings["refractory_samples"] == 30
    assert settings["warmup_samples"] == 3000 and settings["sample_timing"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channels": 0}, "at least one channel"),
        ({"bin_samples": 0}, "bin_samples"),
        ({"bin_samples": 1025}, "bin_samples"),
        ({"power_bins": 65536}, "power_bins"),
        ({"groups": 3}, "groups"),
        ({"sample_rate_hz": 0.0}, "sample_rate_hz"),
        ({"refractory_samples": 19}, "refractory_samples"),
        ({"threshold_sigma": 0.0}, "threshold_sigma"),
        ({"warmup_samples": -1}, "warmup_samples"),
        ({"warmup_samples": (1 << 40) + 1}, "warmup_samples"),
        ({"power_shift": 16}, "power_shift"),
        ({"highpass_hz": 15000.0}, "must lie in"),
        ({"timing": "event"}, "timing"),
        ({"backend": "gpu"}, "backend"),
    ],
)
def test_encoder_settings_are_refused(kwargs: dict[str, Any], message: str) -> None:
    """Every setting outside its documented range is refused before any coding."""
    arguments: dict[str, Any] = {"channels": 2, "sample_rate_hz": RATE, "bin_samples": 20}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SpikeEventEncoder(arguments.pop("channels"), arguments.pop("sample_rate_hz"), **arguments)


def test_decoder_settings_and_windows_are_refused() -> None:
    """Decoder layouts and malformed windows are refused."""
    with pytest.raises(ValueError, match="timing"):
        SpikeEventDecoder(2, 20, timing="event")
    with pytest.raises(ValueError, match="groups"):
        SpikeEventDecoder(2, 20, groups=0)
    encoder = SpikeEventEncoder(2, RATE, backend="python")
    for window, message in [
        (np.zeros((4, 3)), r"\(samples, 2\)"),
        (np.full((2, 2), 0.5), "integer samples"),
        (np.full((2, 2), 40000), "int16 range"),
    ]:
        with pytest.raises(ValueError, match=message):
            encoder.encode(window)
    integral = np.full((40, 2), 3.0)
    window = SpikeEventDecoder(2, 20, backend="python").decode(encoder.encode(integral), 40)
    assert window.events.shape == (2, 2) and not window.events.any()


def _packet(write: Any) -> bytes:
    encoder = ws._RangeEncoder()
    write(encoder, [1 << 15], [ws._PROB_INIT] * 32, ws._ResidualModel())
    body = encoder.finish()
    return len(body).to_bytes(4, "big") + body


@pytest.mark.parametrize("backend", BACKENDS)
def test_hostile_packets_are_refused_identically(backend: str) -> None:
    """Out-of-bin offsets, markers, long escapes, negative power, overruns and unused bytes."""

    def offset_beyond_bin(encoder: Any, probs: Any, tree: Any, power: Any) -> None:
        encoder.bit_precise(probs, 0, 1, se.EVENT_ADAPT_SHIFT)
        se._put_offset(encoder, tree, 25, 32)

    def marker_as_power(encoder: Any, probs: Any, tree: Any, power: Any) -> None:
        encoder.bit_precise(probs, 0, 0, se.EVENT_ADAPT_SHIFT)
        power.put_marker(encoder)

    def escape_too_long(encoder: Any, probs: Any, tree: Any, power: Any) -> None:
        encoder.bit_precise(probs, 0, 0, se.EVENT_ADAPT_SHIFT)
        for node in (1, 3, 7, 15, 31, 63):
            encoder.bit(power.tree, node, 1)
        for index in range(ws.MAX_ESCAPE_BITS):
            encoder.bit(power.escape, index, 1)

    def negative_power(encoder: Any, probs: Any, tree: Any, power: Any) -> None:
        encoder.bit_precise(probs, 0, 0, se.EVENT_ADAPT_SHIFT)
        power.put(encoder, ws._zigzag(-5))

    for write, message in [
        (offset_beyond_bin, "offset beyond its bin"),
        (marker_as_power, "unexpected marker"),
        (escape_too_long, "escape too long"),
        (negative_power, "band power out of range"),
    ]:
        decoder = SpikeEventDecoder(1, 20, timing="sample", power_bins=1, backend=backend)
        with pytest.raises(ValueError, match=message):
            decoder.decode(_packet(write), 20)
        with pytest.raises(ValueError, match="start a new decoder"):
            decoder.decode(_packet(write), 20)
    short = b"\x00" * 5
    with pytest.raises(ValueError, match="ran past its bytes"):
        SpikeEventDecoder(1, 20, power_bins=1, backend=backend).decode(
            len(short).to_bytes(4, "big") + short, 4000
        )
    with pytest.raises(ValueError, match="ran past its bytes"):
        SpikeEventDecoder(1, 20, power_bins=0, backend=backend).decode(
            _packet(
                lambda encoder, probs, tree, power: encoder.bit_precise(
                    probs, 0, 1, se.EVENT_ADAPT_SHIFT
                )
            ),
            4000,
        )
    body = SpikeEventEncoder(1, RATE, backend=backend).encode(np.zeros((40, 1)))[4:]
    padded = body + b"\x00" * 8
    with pytest.raises(ValueError, match="trailing range-coder bytes"):
        SpikeEventDecoder(1, 20, backend=backend).decode(
            len(padded).to_bytes(4, "big") + padded, 40
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_oversized_sample_requests_are_refused_before_allocation(backend: str) -> None:
    """Requests the packet cannot hold are refused; the decoder stays usable."""
    data = _recording(400, [[], []])
    packet = SpikeEventEncoder(2, RATE, power_bins=1, backend=backend).encode(data)
    decoder = SpikeEventDecoder(2, 20, power_bins=1, backend=backend)
    for samples in (1 << 63, 1 << 40, 400 * 4096):
        with pytest.raises(ValueError, match="more samples requested than the packet can hold"):
            decoder.decode(packet, samples)
    assert decoder.decode(packet, 400).power.shape == (20, 2)
    # 400 samples already seen: one more than 2**64 - 400 overflows the sample counter.
    with pytest.raises(ValueError, match="more samples requested"):
        decoder.decode(packet, (1 << 64) - 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_framing_errors_leave_the_decoder_unchanged(backend: str) -> None:
    """Group framing is checked before any state changes."""
    data = _recording(400, [[], []])
    packet = SpikeEventEncoder(2, RATE, groups=2, backend=backend).encode(data)

    def fresh() -> SpikeEventDecoder:
        return SpikeEventDecoder(2, 20, groups=2, backend=backend)

    for broken, message in [
        (packet[:3], "missing group length"),
        (packet[:-1], "longer than packet"),
        (packet + b"\x00", "trailing bytes"),
        (packet[:4] + b"\x01" + packet[5:], "range-coder start"),
    ]:
        with pytest.raises(ValueError, match=message):
            fresh().decode(broken, 400)
    decoder = fresh()
    with pytest.raises(ValueError):
        decoder.decode(packet[:3], 400)
    assert decoder.decode(packet, 400).events.shape == (20, 2)


def test_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the engine ``auto`` uses the reference and ``rust`` is refused."""
    monkeypatch.setitem(sys.modules, "sc_neurocore_engine.sc_neurocore_engine", None)
    assert isinstance(SpikeEventEncoder(1, RATE)._coder, se._PyEncoder)
    assert isinstance(SpikeEventDecoder(1, 20)._coder, se._PyDecoder)
    with pytest.raises(ValueError, match="backend 'rust' needs"):
        SpikeEventDecoder(1, 20, backend="rust")
