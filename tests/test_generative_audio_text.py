# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the SC audio-synthesis and text-generation engines

"""Contracts for the SC audio-synthesis and minimal text-generation engines."""

from __future__ import annotations

import numpy as np

from sc_neurocore.generative.audio_synthesis import SCAudioSynthesizer
from sc_neurocore.generative.text_gen import SCTextGenerator


def test_audio_synthesizer_tone_length_and_amplitude() -> None:
    """synthesize_tone returns a sine buffer of sample-rate length scaled by probability."""
    synth = SCAudioSynthesizer(sample_rate=8000)
    wave = synth.synthesize_tone(frequency=440.0, duration_ms=100, probability=0.5)

    assert wave.shape == (int(8000 * 100 / 1000),)
    assert np.max(np.abs(wave)) <= 0.5 + 1e-9


def test_audio_synthesizer_bitstream_to_audio_preserves_length() -> None:
    """bitstream_to_audio low-pass filters the bitstream while preserving its length."""
    synth = SCAudioSynthesizer()
    bits = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0], dtype=float)

    audio = synth.bitstream_to_audio(bits)

    assert audio.shape == bits.shape


def test_text_generator_token_is_drawn_from_vocabulary() -> None:
    """generate_token selects a token from the configured vocabulary."""
    gen = SCTextGenerator(vocab=["a", "b", "c"])

    token = gen.generate_token(np.array([1.0, 0.0, 0.0]))

    assert token in {"a", "b", "c"}


def test_text_generator_sequence_has_requested_length() -> None:
    """generate_sequence returns a space-joined string with the requested token count."""
    gen = SCTextGenerator(vocab=["x", "y"])

    seq = gen.generate_sequence(length=5)

    assert len(seq.split(" ")) == 5
