# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — WaveformCodec Rust and NumPy backends write identical streams

"""The Rust kernels and the NumPy reference produce byte-identical streams and decodes."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from sc_neurocore.spike_codec import waveform_codec as wc
from sc_neurocore.spike_codec.waveform_codec import WaveformCodec
from tests.waveform_codec_support import _make_waveform

BACKENDS = ("python", "rust")


def _inputs() -> list[tuple[str, np.ndarray[Any, Any]]]:
    """Return named float and int16 inputs that exercise every kernel branch."""
    rng = np.random.default_rng(11)
    shape = np.zeros(48, dtype=np.float32)
    shape[15:20], shape[20:25], shape[25:30] = -200.0, 100.0, -30.0
    scaled = rng.normal(0.0, 1.0, size=(2000, 1)).astype(np.float32)
    for t, amplitude in ((300, 1.0), (900, 2.0), (1500, 0.5)):
        scaled[t - 24 : t + 24, 0] += amplitude * shape
    return [
        (
            "dense spikes, 4 channels",
            _make_waveform(T=4000, N=4, noise_sigma=20.0, spike_rate=80.0),
        ),
        ("noise only, 3 channels", rng.normal(0.0, 50.0, size=(1600, 3)).astype(np.float32)),
        ("scaled shared template", scaled),
        ("short", np.array([[100.0], [101.0], [-400.0], [100.5]], dtype=np.float32)),
        ("odd length", _make_waveform(T=1601, N=3, noise_sigma=20.0, spike_rate=80.0)),
        ("int16 noise", rng.normal(0.0, 400.0, size=(3000, 2)).astype(np.int16).astype(np.float32)),
    ]


@pytest.mark.parametrize("mode", ["spike", "waveform", "full"])
@pytest.mark.parametrize("max_templates", [1, 16])
def test_lossy_streams_and_decodes_are_identical(mode: str, max_templates: int) -> None:
    """Every lossy mode writes the same bytes and decodes to the same arrays on both backends."""
    for name, waveform in _inputs():
        streams = {
            backend: WaveformCodec(
                mode=mode, max_templates=max_templates, backend=backend
            ).compress(waveform)[0]
            for backend in BACKENDS
        }
        assert streams["python"] == streams["rust"], name
        decoded = {
            backend: WaveformCodec.decompress(streams["python"], backend=backend)
            for backend in BACKENDS
        }
        np.testing.assert_array_equal(decoded["python"].spike_raster, decoded["rust"].spike_raster)
        if mode == "spike":
            assert decoded["python"].waveform is None and decoded["rust"].waveform is None
        else:
            np.testing.assert_array_equal(decoded["python"].waveform, decoded["rust"].waveform)


def test_lossless_streams_and_decodes_are_identical() -> None:
    """The lossless mode writes the same bytes and returns the same samples on both backends."""
    rng = np.random.default_rng(12)
    t = np.arange(20000)
    cases = [
        rng.integers(-32768, 32768, size=(1500, 3)).astype(np.int16),
        np.round(np.round(300 * np.sin(t / 50.0)) * 64.0626).astype(np.int16).reshape(-1, 1),
        (rng.integers(-1000, 1000, size=(5000, 1)) * 4 + 3).astype(np.int16),
        np.full((9, 2), -32768, dtype=np.int16),
        np.array([[5]], dtype=np.int16),
    ]
    for samples in cases:
        streams = {
            backend: WaveformCodec(mode="lossless", backend=backend).compress(samples)[0]
            for backend in BACKENDS
        }
        assert streams["python"] == streams["rust"]
        for backend in BACKENDS:
            decoded = WaveformCodec.decompress(streams["python"], backend=backend).waveform
            np.testing.assert_array_equal(decoded, samples)


def test_rust_and_python_refuse_the_same_corrupted_lossless_section() -> None:
    """A sealed stream with an out-of-table level is refused by both decoders."""
    samples = np.round(np.arange(600) % 37 * 64.06).astype(np.int16).reshape(-1, 1)
    body = WaveformCodec(mode="lossless", backend="python").compress(samples)[0][:-4]
    start = 4 + 16 + 4 + 4
    broken = wc._seal(body[:start] + bytes([9]) + body[start + 1 :])
    for backend in BACKENDS:
        with pytest.raises(ValueError, match="level mapping"):
            WaveformCodec.decompress(broken, backend=backend)


def test_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto prefers the installed kernels; a missing engine falls back or refuses explicitly."""
    assert WaveformCodec(backend="rust")._native is not None
    assert WaveformCodec(backend="auto")._native is not None
    assert WaveformCodec(backend="python")._native is None
    with pytest.raises(ValueError, match="backend must be one of"):
        WaveformCodec(backend="gpu")
    monkeypatch.setitem(sys.modules, "sc_neurocore_engine.sc_neurocore_engine", None)
    assert WaveformCodec(backend="auto")._native is None
    with pytest.raises(ValueError, match="backend 'rust' needs"):
        WaveformCodec(backend="rust")


@pytest.mark.parametrize("max_templates", [2, 4, 16])
@pytest.mark.parametrize("threshold", [0.5, 0.9, 0.99])
def test_template_match_is_identical_including_float64_scales(
    max_templates: int, threshold: float
) -> None:
    """Template sources, IDs, float64 scales and residuals agree exactly (correctly rounded dots)."""
    rng = np.random.default_rng(13)
    families = rng.normal(0.0, 1.0, size=(5, 48)).astype(np.float32)
    snippets = [
        (rng.uniform(0.3, 3.0) * families[rng.integers(5)] + rng.normal(0.0, 0.3, 48)).astype(
            np.float32
        )
        for _ in range(300)
    ]
    snippets[7] = np.zeros(48, dtype=np.float32)
    results = {
        backend: WaveformCodec(
            max_templates=max_templates, template_threshold=threshold, backend=backend
        )._template_match(snippets)
        for backend in BACKENDS
    }
    python, rust = results["python"], results["rust"]
    assert len(python[0]) == len(rust[0])
    for a, b in zip(python[0], rust[0], strict=True):
        np.testing.assert_array_equal(a, b)
    assert python[1] == rust[1]
    assert python[2] == rust[2], "float64 scales must agree bit for bit"
    np.testing.assert_array_equal(np.stack(python[3]), np.stack(rust[3]))


def test_background_ties_round_half_to_even(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact half-step deltas are rounded half to even by both quantisers."""
    monkeypatch.setitem(sys.modules, "pywt", None)
    # Largest delta 2 with quantize_bits=2 gives step 2, so a delta of 1 is exactly half a step.
    deltas = np.array([0.0, 1.0, 1.0, -1.0, 2.0, 1.0, -1.0, -2.0, 1.0], dtype=np.float32)
    background = np.cumsum(deltas).astype(np.float32).reshape(-1, 1)
    assert float(wc._background_step(2.0, 2)) == 2.0
    streams = {
        backend: WaveformCodec(quantize_bits=2, backend=backend)._compress_background(background)
        for backend in BACKENDS
    }
    assert streams["python"] == streams["rust"]


def test_rice_parameter_ties_choose_the_smallest() -> None:
    """A +-1 random walk ties two Rice parameters in every block; both coders pick the smaller."""
    rng = np.random.default_rng(14)
    walk = (np.cumsum(rng.choice([-1, 1], size=6000)) + 1000).astype(np.int16).reshape(-1, 1)
    streams = {
        backend: WaveformCodec(mode="lossless", backend=backend).compress(walk)[0]
        for backend in BACKENDS
    }
    header = 4 + 16 + 4 + 4
    assert streams["python"][header] == 0 and streams["python"][header + 9] == 1, (
        "affine mapping, order 1"
    )
    assert streams["python"] == streams["rust"]


@pytest.mark.parametrize("samples", [2000, 2001, 1, 2])
def test_noise_thresholds_are_identical(samples: int) -> None:
    """Median-based thresholds agree bit for bit for even and odd sample counts."""
    rng = np.random.default_rng(15)
    waveform = rng.normal(0.0, 30.0, size=(samples, 5)).astype(np.float32)
    thresholds = {
        backend: WaveformCodec(threshold_sigma=4.5, backend=backend)._noise_thresholds(waveform)
        for backend in BACKENDS
    }
    assert thresholds["python"].dtype == np.float32
    np.testing.assert_array_equal(thresholds["python"], thresholds["rust"])
