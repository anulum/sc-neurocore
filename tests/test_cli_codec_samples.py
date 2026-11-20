# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Codec acquisition and export adapters

"""Run the console with time-major NumPy, PCM and event-table file inputs."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.cli import main
from sc_neurocore.cli.codec.samples import OutputField, write_decoded_samples
from sc_neurocore.spike_codec import WaveformCodec
from tools.installed_codec_cases import acquisition


def test_event_csv_and_partial_windows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A sorted event table recreates the original binary raster and silent tail."""
    raster, _, _ = acquisition(None)
    source, packet, output = (tmp_path / item for item in ("events.csv", "packets", "out.npy"))
    events = [f"{sample},{channel}" for sample, channel in np.argwhere(raster)]
    source.write_text("sample,channel\n" + "\n".join(events) + "\n", encoding="utf-8")
    encode = [
        "codec",
        "encode",
        str(source),
        str(packet),
        "--codec",
        "aer",
        "--frame-rows",
        "97",
    ]
    assert main(encode) == 1
    assert "needs --rows" in json.loads(capsys.readouterr().err)["error"]
    geometry = ["--rows", str(raster.shape[0]), "--channels", str(raster.shape[1])]
    assert main([*encode, *geometry]) == 0
    assert main(["codec", "decode", str(packet), str(output)]) == 0
    np.testing.assert_array_equal(np.load(output, allow_pickle=False), raster)
    with source.open("a", encoding="utf-8") as stream:
        stream.write(events[-1] + "\n")
    saved = packet.read_bytes()
    assert main([*encode, *geometry, "--force"]) == 1
    assert "sorted, unique" in json.loads(capsys.readouterr().err)["error"]
    assert packet.read_bytes() == saved


def test_pcm_waveform_input_and_refusals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stream multichannel PCM16 into a lossless file and refuse a truncated acquisition."""
    _, waveform, rate = acquisition(None)
    source, packet, output = (tmp_path / item for item in ("signal.wav", "packets", "out.npy"))
    with wave.open(str(source), "wb") as writer:
        writer.setparams((waveform.shape[1], 2, rate, len(waveform), "NONE", "uncompressed"))
        writer.writeframes(waveform.astype("<i2").tobytes())
    command = [
        "codec",
        "encode",
        str(source),
        str(packet),
        "--codec",
        "lossless_stream",
        "--backend",
        "python",
        "--frame-rows",
        "127",
        "--option",
        "groups=2",
    ]
    assert main([*command, "--sample-rate", str(rate // 2)]) == 1
    assert "disagrees" in json.loads(capsys.readouterr().err)["error"]
    assert main(command) == 0
    assert main(["codec", "decode", str(packet), str(output)]) == 0
    np.testing.assert_array_equal(np.load(output, allow_pickle=False), waveform)
    source.write_bytes(source.read_bytes()[:-3])
    previous = packet.read_bytes()
    assert main([*command, "--force"]) == 1
    assert "truncated PCM16" in json.loads(capsys.readouterr().err)["error"]
    assert packet.read_bytes() == previous


@pytest.mark.parametrize(
    ("content", "geometry", "codec", "reason"),
    [
        ("invalid\n0,0\n", (20, 3), "aer", "sample,channel header"),
        ("sample,channel\n0,0,1\n", (20, 3), "aer", "one sample and channel"),
        ("sample,channel\n0,0\n", (20, 3), "waveform", "only be encoded by a raster"),
        ("sample,channel\n0,0\n", (0, 3), "aer", "row count"),
        ("sample,channel\n0,0\n", (20, 0), "aer", "channel count"),
    ],
)
def test_event_csv_invalid_contracts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    content: str,
    geometry: tuple[int, int],
    codec: str,
    reason: str,
) -> None:
    """Reject unknown event geometry and malformed CSV without publishing packets."""
    source, packet = tmp_path / "events.csv", tmp_path / "recording"
    source.write_text(content, encoding="utf-8")
    assert (
        main(
            [
                "codec",
                "encode",
                str(source),
                str(packet),
                "--codec",
                codec,
                "--rows",
                str(geometry[0]),
                "--channels",
                str(geometry[1]),
            ]
        )
        == 1
    )
    assert reason in json.loads(capsys.readouterr().err)["error"]
    assert not packet.exists()


def test_sample_file_types_and_implicit_channel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accept a 1-D channel and refuse incompatible shape, format or geometry flags."""
    _, waveform, rate = acquisition(None)
    source, packet, output = (tmp_path / item for item in ("mono.npy", "packets", "out.npy"))
    np.save(source, waveform[:, 0], allow_pickle=False)
    encode = ["codec", "encode", str(source), str(packet), "--codec", "lossless_stream"]
    assert main([*encode, "--rows", str(len(waveform)), "--channels", "1"]) == 1
    assert "apply only to event CSV" in json.loads(capsys.readouterr().err)["error"]
    assert main(encode) == 0
    assert main(["codec", "decode", str(packet), str(output)]) == 0
    np.testing.assert_array_equal(np.load(output, allow_pickle=False), waveform[:, :1])
    np.save(source, waveform[:, :, None], allow_pickle=False)
    assert main([*encode, "--force"]) == 1
    assert "one- or two-dimensional" in json.loads(capsys.readouterr().err)["error"]

    pcm8 = tmp_path / "pcm8.wav"
    with wave.open(str(pcm8), "wb") as stream:
        stream.setparams((1, 1, rate, len(waveform), "NONE", "uncompressed"))
        stream.writeframes((waveform[:, 0] >> 8).astype(np.uint8).tobytes())
    assert main(["codec", "encode", str(pcm8), str(packet), "--codec", "waveform", "--force"]) == 1
    assert "signed 16-bit samples" in json.loads(capsys.readouterr().err)["error"]
    unknown = source.with_suffix(".dat")
    source.rename(unknown)
    assert (
        main(["codec", "encode", str(unknown), str(packet), "--codec", "waveform", "--force"]) == 1
    )
    assert "sample input must be" in json.loads(capsys.readouterr().err)["error"]


@pytest.mark.parametrize(
    ("rows", "channels", "reason"),
    [
        (1027, 2, "incompatible channel layout"),
        (1026, 3, "exceeds its declared row count"),
        (1028, 3, "incomplete row count"),
    ],
)
def test_decoded_array_writer_checks_committed_layout(
    tmp_path: Path, rows: int, channels: int, reason: str
) -> None:
    """The NPY export sink refuses original decoded samples with a mismatched array layout."""
    _, waveform, _ = acquisition(None)
    encoder = WaveformCodec(mode="lossless", backend="python")
    packet = encoder.compress(waveform)[0]
    restored = WaveformCodec.decompress(packet, backend="python").waveform
    assert restored is not None
    field = OutputField("waveform", rows, channels, "<i2")
    with pytest.raises(ValueError, match=reason):
        write_decoded_samples(
            tmp_path / "unpublished.npy", (field,), ({"waveform": restored},), archive=False
        )
