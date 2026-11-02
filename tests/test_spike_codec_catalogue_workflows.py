# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real codec workflows discovered through the catalogue

"""Exercise public raster, waveform and stateful interfaces with their distinct contracts."""

from __future__ import annotations

from importlib import import_module
import struct
from typing import Any
import zlib

import numpy as np
import pytest

from sc_neurocore.spike_codec import describe_codec, get_codec, probe_codec


def _entrypoint(path: str) -> Any:
    module, _, symbol = path.rpartition(".")
    return getattr(import_module(module), symbol)


@pytest.mark.parametrize("name", ["isi", "predictive", "delta", "streaming", "aer"])
def test_raster_public_factory_roundtrips_described_variants(name: str) -> None:
    """Every available predictor and lossless raster family retains original data and packet bytes."""
    descriptor = describe_codec(name)
    raster = np.zeros((257, 9), dtype=np.int8)
    raster[3::23, :] = 1
    raster[0, 0] = raster[-1, -1] = 1
    for variant in descriptor.variants:
        if variant.name == "lossy":
            continue
        availability = probe_codec(name, variant.name)
        if not availability.python_available:
            assert variant.name == "world_model" and availability.diagnostics
            continue
        kwargs = (
            {descriptor.variant_parameter: variant.name} if descriptor.variant_parameter else {}
        )
        encoder = get_codec(name, **kwargs)
        packet, result = encoder.compress(raster)
        decoder = _entrypoint(descriptor.decoder_entrypoint)()
        np.testing.assert_array_equal(decoder.decompress(packet, *raster.shape), raster)
        assert result.lossless
        describe_codec(name)
        probe_codec(name, variant.name)
        assert encoder.compress(raster)[0] == packet
        for wire in descriptor.wire_formats:
            if wire.magic_hex is not None and packet.startswith(bytes.fromhex(wire.magic_hex)):
                break
        else:
            assert name == "isi" and packet[:4] == struct.pack("!I", len(raster))


@pytest.mark.parametrize("entropy", ["auto", "varint", "huffman"])
def test_isi_entropy_and_lossy_bin_geometry_are_explicit(entropy: str) -> None:
    """Quantization returns bin indices and drops an incomplete input bin, without time expansion."""
    raster = np.zeros((17, 2), dtype=np.int8)
    raster[[0, 3, 7, 16], 0] = 1
    raster[[2, 12], 1] = 1
    codec = get_codec("isi", mode="lossy", timing_precision=4, entropy=entropy)
    packet, result = codec.compress(raster)
    expected = np.zeros_like(raster)
    expected[[0, 1], 0] = 1
    expected[[0, 3], 1] = 1
    np.testing.assert_array_equal(codec.decompress(packet, *raster.shape), expected)
    assert not result.lossless
    lossless = get_codec("isi", entropy=entropy)
    original_packet, _ = lossless.compress(raster)
    np.testing.assert_array_equal(lossless.decompress(original_packet, *raster.shape), raster)
    if entropy == "huffman":
        assert original_packet[:1] == b"\x01"
    if entropy == "varint":
        assert original_packet[:8] == struct.pack("!II", *raster.shape)


@pytest.mark.parametrize(
    "mode,mode_byte",
    [("full", 0), ("waveform", 1), ("spike", 2), ("lossless_rice", 3), ("lossless", 4)],
)
def test_waveform_modes_cross_decode_and_match_wfc2_header(mode: str, mode_byte: int) -> None:
    """Actual WFC2 bytes, CRC and decoded result kinds match the catalogue in every mode."""
    descriptor = describe_codec("waveform")
    cls = _entrypoint(descriptor.encoder_entrypoint)
    rng = np.random.default_rng(13)
    samples = rng.integers(-30, 31, (800, 2)).astype(np.int16)
    samples[400:405, 0] = [-2000, -6000, -3000, 1000, 500]
    packets = []
    availability = probe_codec("waveform", mode)
    backends = [choice for choice in availability.backend_choices if choice != "auto"]
    assert "python" in backends
    for backend in backends:
        packet, result = cls(mode=mode, backend=backend).compress(samples)
        packets.append(packet)
        assert packet[:4] == b"WFC2"
        header = struct.unpack("!IIHHBBBB", packet[4:20])
        assert header[:2] == samples.shape and header[-1] == mode_byte
        assert struct.unpack("!I", packet[-4:])[0] == zlib.crc32(packet[:-4])
        assert result.compressed_bytes == len(packet)
        for decoder_backend in backends:
            decoded = cls.decompress(packet, backend=decoder_backend)
            assert decoded.mode == mode
            assert decoded.lossless_waveform == (descriptor.variant(mode).fidelity == "lossless")
            if decoded.lossless_waveform:
                np.testing.assert_array_equal(decoded.waveform, samples)
                assert decoded.waveform.dtype == np.int16
            else:
                assert decoded.spike_raster.shape == samples.shape
                assert int(decoded.spike_raster.sum()) == result.n_spikes_detected
                if mode == "spike":
                    assert decoded.waveform is None
                else:
                    assert decoded.waveform.shape == samples.shape
    assert all(packet == packets[0] for packet in packets)
    corrupt = bytearray(packets[0])
    corrupt[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        cls.decompress(bytes(corrupt), backend="python")


def test_lossless_stream_public_state_and_tail_cross_decode() -> None:
    """Metadata supplies matching constructors; packets preserve state and a short final window."""
    descriptor = describe_codec("lossless_stream")
    encoder_cls = _entrypoint(descriptor.encoder_entrypoint)
    decoder_cls = _entrypoint(descriptor.decoder_entrypoint)
    samples = (np.arange(257 * 3).reshape(257, 3) % 47 - 23).astype(np.int16)
    backends = [b for b in probe_codec("lossless_stream").backend_choices if b != "auto"]
    packets = {}
    windows = [samples[i : i + 20] for i in range(0, len(samples), 20)]
    for backend in backends:
        encoder = encoder_cls(3, groups=2, backend=backend)
        packets[backend] = [encoder.encode(window) for window in windows]
    for backend in backends:
        for encoded in packets.values():
            decoder = decoder_cls(3, groups=2, backend=backend)
            rows = [
                decoder.decode(packet, len(window))
                for packet, window in zip(encoded, windows, strict=True)
            ]
            np.testing.assert_array_equal(np.concatenate(rows), samples)
    assert all(items == packets["python"] for items in packets.values())
    assert len(windows[-1]) == 17
    assert not descriptor.wire_formats[0].dimensions_in_header
    assert "preceding_state" in descriptor.wire_formats[0].required_decode_metadata


@pytest.mark.parametrize("timing", ["bin", "sample"])
def test_event_stream_public_settings_and_decoded_data(timing: str) -> None:
    """Both timing modes cross-decode real detected events/power, including carried partial bins."""
    descriptor = describe_codec("spike_events")
    encoder_cls = _entrypoint(descriptor.encoder_entrypoint)
    decoder_cls = _entrypoint(descriptor.decoder_entrypoint)
    rng = np.random.default_rng(14)
    samples = rng.integers(-100, 101, (801, 2)).astype(np.int16)
    samples[400:409, 0] = [-3000, -9000, -12000, -7000, -2000, 1500, 2500, 2000, 1000]
    windows = [samples[i : i + 37] for i in range(0, len(samples), 37)]
    backends = [b for b in probe_codec("spike_events", timing).backend_choices if b != "auto"]
    packets = {}
    for backend in backends:
        encoder = encoder_cls(
            2, 19531.0, timing=timing, warmup_samples=100, power_bins=2, groups=2, backend=backend
        )
        packets[backend] = [encoder.encode(window) for window in windows]
        assert encoder.settings["bin_samples"] == 20
    assert all(items == packets["python"] for items in packets.values())
    reference = None
    for backend in backends:
        decoder = decoder_cls(2, 20, timing=timing, power_bins=2, groups=2, backend=backend)
        decoded = [
            decoder.decode(packet, len(window))
            for packet, window in zip(packets["python"], windows, strict=True)
        ]
        events = np.concatenate([row.events for row in decoded])
        power = np.concatenate([row.power for row in decoded])
        assert events.shape == (40, 2) and events.dtype == np.uint8
        assert power.shape == (20, 2) and power.dtype == np.int32
        assert events[:, 0].sum() > 0
        assert any(row.first_bin > 0 for row in decoded)
        if timing == "sample":
            offsets = np.concatenate([row.offsets for row in decoded])
            assert offsets.dtype == np.int16 and np.all(offsets[events > 0] < 20)
        else:
            assert all(row.offsets is None for row in decoded)
        if reference is None:
            reference = events, power
        else:
            np.testing.assert_array_equal(events, reference[0])
            np.testing.assert_array_equal(power, reference[1])
