# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Import of legacy BCI spike recordings

"""Exercise migration of legacy BCI bytes through the real codec console."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.bci_studio.bci_studio import SpikeCodec
from sc_neurocore.cli import main
from sc_neurocore.spike_codec import probe_codec
from tools.installed_codec_cases import acquisition


def test_cli_imports_legacy_bci_telemetry_with_explicit_geometry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve time and channel values across legacy runs and framed packets."""
    raster, _, rate = acquisition(None)
    raster[:301] = 0
    source, packet, output = (
        tmp_path / "telemetry.bin",
        tmp_path / "recording.sc-codec",
        tmp_path / "restored.npy",
    )
    legacy = SpikeCodec()
    payload = legacy.encode(raster.reshape(-1))
    source.write_bytes(payload)
    native = "rust" in probe_codec("aer").backend_choices
    source_args = [
        "--legacy-rle",
        "--codec",
        "aer",
        "--rows",
        str(raster.shape[0]),
        "--channels",
        str(raster.shape[1]),
        "--sample-rate",
        str(rate),
        "--frame-rows",
        "127",
    ]
    assert (
        main(["codec", "encode", str(source), str(packet), *source_args, "--backend", "python"])
        == 0
    )
    encoded = json.loads(capsys.readouterr().out)
    assert encoded["file"]["rows"] == raster.shape[0]
    assert encoded["file"]["channels"] == raster.shape[1]
    assert encoded["file"]["sample_rate_hz"] == rate
    assert (
        main(
            [
                "codec",
                "decode",
                str(packet),
                str(output),
                "--backend",
                "rust" if native else "python",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["verified"]
    restored = np.load(output, allow_pickle=False)
    np.testing.assert_array_equal(restored, raster)
    np.testing.assert_array_equal(legacy.decode(source.read_bytes()), raster.reshape(-1))
    assert source.read_bytes() == payload

    assert (
        main(
            [
                "codec",
                "benchmark",
                str(source),
                *source_args,
                "--repetitions",
                "1",
                "--backend",
                "compare" if native else "python",
            ]
        )
        == 0
    )
    runs = json.loads(capsys.readouterr().out)["runs"]
    assert {run["backend"] for run in runs} == ({"python", "rust"} if native else {"python"})
    assert len({run["packet_sha256"] for run in runs}) == 1
    assert len({run["decoded_sha256"] for run in runs}) == 1


@pytest.mark.parametrize(
    ("declared", "runs", "geometry", "codec", "reason"),
    [
        (None, b"", ("1", "3"), "isi", "length does not match"),
        (4, b"\x00\x04", ("1", "3"), "isi", "length does not match"),
        (3, b"\x00\x03", ("1", "0"), "isi", "positive rows and channels"),
        (3, b"\x00\x03", None, "isi", "positive rows and channels"),
        (3, b"\x00\x01\x01", ("1", "3"), "isi", "truncated legacy RLE run"),
        (3, b"\x02\x03", ("1", "3"), "isi", "invalid legacy RLE run"),
        (3, b"\x01\x00", ("1", "3"), "isi", "invalid legacy RLE run"),
        (3, b"\x01\x04", ("1", "3"), "isi", "invalid legacy RLE run"),
        (3, b"\x00\x03\x01\x01", ("1", "3"), "isi", "do not match the declared raster"),
        (3, b"\x00\x03", ("1", "3"), "lossless_stream", "only be encoded by a raster codec"),
    ],
)
def test_cli_rejects_invalid_legacy_recordings_atomically(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    declared: int | None,
    runs: bytes,
    geometry: tuple[str, str] | None,
    codec: str,
    reason: str,
) -> None:
    """Refuse malformed legacy streams and retain a pre-existing codec recording."""
    source, packet = tmp_path / "telemetry.bin", tmp_path / "recording.sc-codec"
    source.write_bytes(SpikeCodec().encode(np.array([0, 1, 0], dtype=np.uint8)))
    command = ["codec", "encode", str(source), str(packet), "--legacy-rle", "--backend", "python"]
    assert main([*command, "--codec", "isi", "--rows", "1", "--channels", "3"]) == 0
    capsys.readouterr()
    previous = packet.read_bytes()
    source.write_bytes((struct.pack("<I", declared) if declared is not None else b"") + runs)
    arguments = [] if geometry is None else ["--rows", geometry[0], "--channels", geometry[1]]
    assert main([*command, "--codec", codec, *arguments, "--force"]) == 1
    assert reason in json.loads(capsys.readouterr().err)["error"]
    assert packet.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == [packet.name, source.name]
