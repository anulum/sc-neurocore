# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lossless packet limits refuse unsafe input without changing valid bytes

"""Public regressions for ambiguous ISI headers and overflowing delta reference indices."""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import numpy as np
import pytest

from sc_neurocore.spike_codec import DeltaSpikeCodec, SpikeCodec, get_codec


@pytest.mark.parametrize("entropy", ["varint", "auto"])
@pytest.mark.parametrize("timesteps", [2**24, 2**25])
def test_varint_refuses_unsafe_header_range(entropy: str, timesteps: int) -> None:
    """The current raw header must refuse long rasters instead of being mistaken for Huffman."""
    # No channels: exercise the real dimension boundary without a large allocation.
    raster = np.zeros((timesteps, 0), dtype=np.int8)
    with pytest.raises(ValueError, match=r"varint packets require T < 2\*\*24"):
        SpikeCodec(entropy=entropy).compress(raster)


def test_last_safe_varint_header_and_explicit_long_huffman() -> None:
    """Keep the valid boundary byte-identical and allow explicitly unambiguous Huffman packets."""
    raster = np.zeros((2**24 - 1, 0), dtype=np.int8)
    codec = SpikeCodec(entropy="varint")
    packet, _ = codec.compress(raster)
    assert packet == struct.pack("!II", *raster.shape)
    assert codec.decompress(packet, *raster.shape).shape == raster.shape
    raster = np.zeros((2**24, 0), dtype=np.int8)
    codec = SpikeCodec(entropy="huffman")
    packet, _ = codec.compress(raster)
    assert packet[:9] == b"\x01" + struct.pack("!II", *raster.shape)
    assert codec.decompress(packet, *raster.shape).shape == raster.shape


def test_varint_refuses_real_mono_raster_at_ambiguous_boundary() -> None:
    """The original nonempty-channel reproducer must fail before emitting an ambiguous packet."""
    raster = np.zeros((2**24, 1), dtype=np.int8)
    raster[-1, 0] = 1
    with pytest.raises(ValueError, match="Huffman marker collision"):
        SpikeCodec(entropy="varint").compress(raster)


@pytest.mark.parametrize("group_size", [-1, 0, 257, 65535])
def test_delta_refuses_group_sizes_outside_reference_width(group_size: int) -> None:
    """Group-local references must fit uint8 rather than wrap silently during serialization."""
    with pytest.raises(ValueError, match="group_size must be in 1..256"):
        DeltaSpikeCodec(group_size=group_size)


def test_delta_rechecks_mutated_configuration_before_encoding() -> None:
    """A later configuration change must not bypass refusal at the encode boundary."""
    raster = np.zeros((4, 257), dtype=np.int8)
    raster[:, 256] = 1
    codec = DeltaSpikeCodec()
    codec.group_size = 257
    with pytest.raises(ValueError, match="group_size must be in 1..256"):
        codec.compress(raster)


def test_delta_maximum_reference_and_large_total_channel_count_roundtrip() -> None:
    """Local reference255 works across multiple groups and a final partial group."""
    raster = np.zeros((8, 513), dtype=np.int8)
    raster[:, 255] = raster[:, 511] = 1
    raster[-1, 512] = 1
    codec = DeltaSpikeCodec(group_size=256)
    packet, result = codec.compress(raster)
    assert packet[:8] == b"DSCX" + struct.pack("!HH", 256, 3)
    assert packet[8:11] == bytes([255, 255, 0])
    np.testing.assert_array_equal(codec.decompress(packet, *raster.shape), raster)
    assert result.lossless


GOLDENS = [
    (
        "isi",
        {"entropy": "varint"},
        "41799679ba57daadf4dc36057c625a1bb98eb5f9cea491cbda88e68cddbc8345",
    ),
    (
        "isi",
        {"entropy": "huffman"},
        "e92d71a07d45bdc482b64c73eddfb15f5b44a2386a5a26e293a34cba6d52d815",
    ),
    (
        "isi",
        {"entropy": "auto"},
        "e92d71a07d45bdc482b64c73eddfb15f5b44a2386a5a26e293a34cba6d52d815",
    ),
    (
        "delta",
        {"group_size": 4},
        "e600a100e10b241bc30b4731906374bd17c9c2a3ab5fcf7deeb52229606e4640",
    ),
    (
        "predictive",
        {"predictor": "ema"},
        "c4f59c1008988fe32e834d069b326ae358058591ddb1f027c30d02b0ec1e1cfa",
    ),
    (
        "predictive",
        {"predictor": "lfsr"},
        "54a619052f5291a6a65d1f86af743a8cfcdc759e24f10a9c023d082034d95a86",
    ),
    (
        "predictive",
        {"predictor": "context"},
        "ce1937c26bf567aae80f30583d5dffd876ee28903b76284bf99bfa9718d1c8e7",
    ),
    ("streaming", {}, "058cb6f58fd94773dda3743bf98c5b92037064efc1f80ff4905784df074a4d08"),
    ("aer", {}, "c24887670d03d94ca6d01e6072e94d37f7df9bc6597f6f5d34762dc665667d90"),
]


@pytest.mark.parametrize(("name", "kwargs", "expected_sha256"), GOLDENS)
def test_valid_packets_match_qualified_pre_guard_normal_wheel(
    name: str, kwargs: dict[str, Any], expected_sha256: str
) -> None:
    """Keep exact previously emitted bytes across entropy modes and all raster families."""
    raster = np.zeros((41, 9), dtype=np.int8)
    raster[3::7, :] = 1
    raster[0, 0] = raster[-1, -1] = 1
    codec = get_codec(name, **kwargs)
    packet, result = codec.compress(raster)
    assert hashlib.sha256(packet).hexdigest() == expected_sha256
    assert result.lossless
    np.testing.assert_array_equal(codec.decompress(packet, *raster.shape), raster)
