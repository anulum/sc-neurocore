# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Codec file integrity and publication

"""Exercise framed file integrity and transactional output through the console."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sc_neurocore
from sc_neurocore.cli import main
from tools.installed_codec_cases import acquisition


@pytest.mark.parametrize("damage", ("truncated", "checksum", "suffix", "magic"))
def test_refused_packet_keeps_existing_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    """Refuse malformed files after real encoding without publishing a partial decode."""
    raster, _, _ = acquisition(None)
    source, packet, output = (tmp_path / item for item in ("in.npy", "packets", "out.npy"))
    np.save(source, raster, allow_pickle=False)
    assert main(["codec", "encode", str(source), str(packet), "--codec", "isi"]) == 0
    original = packet.read_bytes()
    changed = {
        "truncated": original[:-33],
        "checksum": original[:-1] + bytes([original[-1] ^ 1]),
        "suffix": original + b"trailing frame data",
        "magic": b"\x00" + original[1:],
    }[damage]
    packet.write_bytes(changed)
    np.save(output, raster[:23], allow_pickle=False)
    previous = output.read_bytes()
    assert main(["codec", "decode", str(packet), str(output), "--force"]) == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert {
        "truncated": "truncated",
        "checksum": "checksum mismatch",
        "suffix": "trailing bytes",
        "magic": "version",
    }[damage] in error
    assert output.read_bytes() == previous
    assert not tuple(tmp_path.glob(".sc-neurocore-codec-*"))


def test_console_refuses_overwrite_and_alias(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve an existing recording until an explicit successful replacement."""
    raster, _, _ = acquisition(None)
    source = tmp_path / "source.npy"
    packet = tmp_path / "encoded"
    np.save(source, raster, allow_pickle=False)
    packet.write_bytes(b"previous recording")
    command = ["codec", "encode", str(source), str(packet), "--codec", "aer"]
    assert main(command) == 1
    assert "already exists" in json.loads(capsys.readouterr().err)["error"]
    assert packet.read_bytes() == b"previous recording"
    assert main([*command, "--force"]) == 0
    assert packet.read_bytes() != b"previous recording"
    assert main(["codec", "encode", str(source), str(source), "--codec", "aer", "--force"]) == 1
    assert "same file" in json.loads(capsys.readouterr().err)["error"]
    np.testing.assert_array_equal(np.load(source, allow_pickle=False), raster)


@pytest.mark.parametrize(
    ("metadata", "packet_size", "reason"),
    [
        (b"", None, "header size"),
        (b"{}", None, "metadata fields"),
        (None, 0, "packet size"),
    ],
)
def test_codec_file_rejects_invalid_framing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    metadata: bytes | None,
    packet_size: int | None,
    reason: str,
) -> None:
    """Reject an incomplete metadata object or invalid frame length before decoding."""
    raster, _, _ = acquisition(None)
    source, packet = tmp_path / "source.npy", tmp_path / "encoded"
    np.save(source, raster, allow_pickle=False)
    assert main(["codec", "encode", str(source), str(packet), "--codec", "isi"]) == 0
    data = packet.read_bytes()
    offset = 13 + struct.unpack_from("!I", data, 9)[0]
    if metadata is not None:
        packet.write_bytes(data[:9] + struct.pack("!I", len(metadata)) + metadata + data[offset:])
    elif packet_size is not None:
        packet.write_bytes(data[:offset] + struct.pack("!I", packet_size) + data[offset + 4 :])
    assert main(["codec", "verify", str(packet)]) == 1
    assert reason in json.loads(capsys.readouterr().err)["error"]


@pytest.mark.parametrize(
    ("codec", "replacement", "reason"),
    [
        ("aer", {"options": {"groups": []}}, "metadata types"),
        ("aer", {"channels": 2}, "geometry disagrees"),
        ("waveform", {"variant": "spike"}, "mode disagrees"),
        ("waveform", {"channels": 1, "rows": 3081, "frame_rows": 3081}, "incompatible dimensions"),
    ],
)
def test_codec_file_refuses_metadata_that_disagrees_with_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    codec: str,
    replacement: dict[str, object],
    reason: str,
) -> None:
    """A valid checksum cannot substitute for agreement between metadata and packet."""
    raster, waveform, _ = acquisition(None)
    source, packet, output = (tmp_path / item for item in ("source.npy", "packets", "out.npz"))
    np.save(source, raster if codec == "aer" else waveform, allow_pickle=False)
    assert main(["codec", "encode", str(source), str(packet), "--codec", codec]) == 0
    data = packet.read_bytes()
    offset = 13 + struct.unpack_from("!I", data, 9)[0]
    header = json.loads(data[13:offset])
    header.update(replacement)
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    amended = data[:9] + struct.pack("!I", len(payload)) + payload + data[offset:-32]
    packet.write_bytes(amended + hashlib.sha256(amended).digest())
    assert main(["codec", "decode", str(packet), str(output)]) == 1
    assert reason in json.loads(capsys.readouterr().err)["error"]
    assert not output.exists()


def test_cancelled_console_write_preserves_destination(tmp_path: Path) -> None:
    """A real interrupt after staging starts must leave the old recording intact."""
    _, waveform, _ = acquisition(None)
    source, output = tmp_path / "input.npy", tmp_path / "encoded"
    np.save(source, np.tile(waveform[:, :1], (40, 1)), allow_pickle=False)
    output.write_bytes(b"previous recording")
    package = Path(sc_neurocore.__file__).resolve().parent.parent
    child = """import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from sc_neurocore.cli import main

output = Path(sys.argv[3])
def interrupt_at_staging():
    while not tuple(output.parent.glob(".sc-neurocore-codec-*/output")):
        threading.Event().wait(0.01)
    signal.raise_signal(signal.SIGINT)

threading.Thread(target=interrupt_at_staging, daemon=True).start()
raise SystemExit(main([
    "codec", "encode", sys.argv[2], sys.argv[3], "--codec", "lossless_stream",
    "--frame-rows", "4096", "--backend", "python", "--force",
]))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", child, str(package), str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 130, result.stderr
    assert json.loads(result.stderr)["error"] == "codec command interrupted"
    assert output.read_bytes() == b"previous recording"
    assert not tuple(tmp_path.glob(".sc-neurocore-codec-*"))
