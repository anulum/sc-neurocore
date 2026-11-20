# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Codec options and runtime routing

"""Verify options and decoded fields through the real console command surface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.cli import main
from sc_neurocore.spike_codec import SpikeEventDecoder, SpikeEventEncoder
from tools.installed_codec_cases import acquisition


def test_event_field_selection_retains_exact_stream_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Compare the exported offsets and power with independently driven endpoints."""
    _, waveform, rate = acquisition(None)
    source, packet, detached = (
        tmp_path / item for item in ("source.npy", "packets", "offsets.npy")
    )
    np.save(source, waveform, allow_pickle=False)
    assert (
        main(
            [
                "codec",
                "encode",
                str(source),
                str(packet),
                "--codec",
                "spike_events",
                "--variant",
                "sample",
                "--frame-rows",
                "103",
                "--sample-rate",
                str(rate),
                "--option",
                "groups=2",
                "--option",
                "bin_samples=17",
                "--option",
                "power_bins=3",
            ]
        )
        == 0
    )
    assert main(["codec", "decode", str(packet), str(detached)]) == 1
    assert "multiple fields" in json.loads(capsys.readouterr().err)["error"]
    assert main(["codec", "decode", str(packet), str(detached), "--field", "offsets"]) == 0
    encoder = SpikeEventEncoder(
        waveform.shape[1],
        rate,
        timing="sample",
        groups=2,
        power_bins=3,
        bin_samples=17,
        backend="python",
    )
    decoder = SpikeEventDecoder(
        waveform.shape[1], 17, timing="sample", groups=2, power_bins=3, backend="python"
    )
    windows = (waveform[index : index + 103] for index in range(0, len(waveform), 103))
    expected = [decoder.decode(encoder.encode(window), len(window)).offsets for window in windows]
    values = [field for field in expected if field is not None]
    assert len(values) == len(expected)
    np.testing.assert_array_equal(np.load(detached, allow_pickle=False), np.concatenate(values))
    assert main(["codec", "decode", str(packet), str(detached), "--field", "waveform"]) == 1
    assert "no 'waveform' field" in json.loads(capsys.readouterr().err)["error"]


@pytest.mark.parametrize(
    ("options", "error"),
    [
        (["--option", "timing_precision=3"], "lossless rasters require"),
        (["--option", "timing_precision=0"], "must be a positive integer"),
        (["--option", "entropy=unknown"], "entropy must be"),
        (["--option", "adc_bits=16"], "options unavailable"),
        (["--option", "entropy"], "unique NAME=VALUE"),
        (["--frame-rows", "0"], "row or sample limit"),
        (["--sample-rate", "0"], "sample rate must be"),
        (["--backend", "gpu"], "cannot use backend"),
    ],
)
def test_incompatible_options_do_not_publish(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], options: list[str], error: str
) -> None:
    """Refuse inconsistent settings before a final packet file exists."""
    raster, _, _ = acquisition(None)
    source, destination = tmp_path / "source.npy", tmp_path / "encoded"
    np.save(source, raster, allow_pickle=False)
    assert main(["codec", "encode", str(source), str(destination), "--codec", "isi", *options]) == 1
    assert error in json.loads(capsys.readouterr().err)["error"]
    assert not destination.exists()


def test_predictive_lossy_frames_use_local_bins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI preserves per-packet quantisation and the original raster dimensions."""
    raster, _, _ = acquisition(None)
    source, packet, output = (tmp_path / item for item in ("source.npy", "packets", "out.npy"))
    np.save(source, raster, allow_pickle=False)
    assert (
        main(
            [
                "codec",
                "encode",
                str(source),
                str(packet),
                "--codec",
                "predictive",
                "--variant",
                "ema",
                "--frame-rows",
                "103",
                "--option",
                "timing_precision=4",
            ]
        )
        == 0
    )
    assert main(["codec", "decode", str(packet), str(output)]) == 0
    restored = np.load(output, allow_pickle=False)
    expected = np.zeros_like(raster)
    for start in range(0, len(raster), 103):
        complete = min(103, len(raster) - start) // 4
        expected[start : start + complete] = (
            raster[start : start + complete * 4].reshape(complete, 4, raster.shape[1]).max(axis=1)
        )
    np.testing.assert_array_equal(restored, expected)
    assert main(["codec", "inspect", "--file", str(packet), "--variant", "ema"]) == 1
    assert "only valid" in json.loads(capsys.readouterr().err)["error"]


@pytest.mark.parametrize(
    ("codec", "options", "reason"),
    [
        ("spike_events", [], "requires the input sample rate"),
        ("streaming", ["--option", "window_size=65536"], "window exceeds"),
        ("waveform", ["--variant", "full", "--option", "adc_bits=16"], "only in lossless modes"),
    ],
)
def test_rejects_incompatible_file_route(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    codec: str,
    options: list[str],
    reason: str,
) -> None:
    """Explicit geometry and mode violations produce an authored CLI error."""
    raster, _, _ = acquisition(None)
    source, packet = tmp_path / "source.npy", tmp_path / "recording"
    np.save(source, raster, allow_pickle=False)
    assert main(["codec", "encode", str(source), str(packet), "--codec", codec, *options]) == 1
    assert reason in json.loads(capsys.readouterr().err)["error"]
    assert not packet.exists()
