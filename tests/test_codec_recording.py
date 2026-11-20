# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Recording codec consumer contract

"""Exercise recording service callbacks, file interoperability and cancellation."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.cli import main
from sc_neurocore.cli.codec.recording import RecordingOptions, decode_recording, encode_recording
from sc_neurocore.cli.codec.samples import Samples
from tools.installed_codec_cases import acquisition


def test_recording_callbacks_preserve_partial_windows_and_console_exports(tmp_path: Path) -> None:
    """Observe real input offsets and export a service recording through the console."""
    _, waveform, rate = acquisition(None)
    source = tmp_path / "acquisition.npy"
    packet = tmp_path / "recording.sc-codec"
    exported = tmp_path / "export.npy"
    np.save(source, waveform, allow_pickle=False)

    expected_starts = list(range(0, len(waveform), 257))
    expected_progress = [
        (min(start + 257, len(waveform)), len(waveform)) for start in expected_starts
    ]
    encoded_rows: list[tuple[int, int]] = []
    decoded_rows: list[tuple[int, int]] = []
    preview_starts: list[int] = []

    def preview(start: int, fields: Mapping[str, Samples]) -> None:
        preview_starts.append(start)
        values = fields["waveform"]
        assert not values.flags.writeable
        np.testing.assert_array_equal(values, waveform[start : start + len(values)])

    encoded = encode_recording(
        source,
        packet,
        RecordingOptions("lossless_stream", frame_rows=257, sample_rate_hz=rate),
        backend="python",
        progress=lambda current, total: encoded_rows.append((current, total)),
    )
    decoded = decode_recording(
        packet,
        backend="python",
        progress=lambda current, total: decoded_rows.append((current, total)),
        preview=preview,
    )
    assert decoded.header == encoded.header
    assert encoded.size_bytes == packet.stat().st_size
    assert encoded_rows == decoded_rows == expected_progress
    assert preview_starts == expected_starts
    assert main(["codec", "decode", str(packet), str(exported), "--backend", "python"]) == 0
    np.testing.assert_array_equal(np.load(exported, allow_pickle=False), waveform)


@pytest.mark.parametrize("operation", ["encode", "decode"])
def test_recording_cancellation_keeps_the_previous_destination(
    tmp_path: Path, operation: str
) -> None:
    """Abort actual window processing and retain an existing valid destination."""
    _, waveform, rate = acquisition(None)
    source = tmp_path / "acquisition.npy"
    packet = tmp_path / "recording.sc-codec"
    exported = tmp_path / "export.npy"
    options = RecordingOptions("lossless_stream", frame_rows=257, sample_rate_hz=rate)
    np.save(source, waveform, allow_pickle=False)
    encode_recording(source, packet, options, backend="python")
    decode_recording(packet, exported, backend="python")
    destination = packet if operation == "encode" else exported
    previous = destination.read_bytes()

    def cancel(current: int, total: int) -> None:
        assert current == 257 and total == len(waveform)
        raise InterruptedError("recording cancelled")

    with pytest.raises(InterruptedError, match="recording cancelled"):
        if operation == "encode":
            encode_recording(
                source, packet, options, backend="python", overwrite=True, progress=cancel
            )
        else:
            decode_recording(packet, exported, backend="python", overwrite=True, progress=cancel)
    assert destination.read_bytes() == previous
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        source.name,
        exported.name,
        packet.name,
    ]
