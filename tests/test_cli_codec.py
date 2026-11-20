# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Codec console integration

"""Exercise the actual console dispatcher on recorded and simulated file inputs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.cli import main
from sc_neurocore.cli.codec import samples
from sc_neurocore.spike_codec import list_codec_descriptors, probe_codec
from tools.installed_codec_cases import acquisition

_VARIANTS = tuple(
    (description.name, variant.name, description.input_kind)
    for description in list_codec_descriptors()
    for variant in description.variants
)


def test_console_discovery_and_inspection(capsys: pytest.CaptureFixture[str]) -> None:
    """The real parser resolves every installed catalogue variant and inspection."""
    assert main(["codec", "discover"]) == 0
    discovered = json.loads(capsys.readouterr().out)["codecs"]
    assert {item["name"] for item in discovered} == {family for family, _, _ in _VARIANTS}
    assert main(["codec", "inspect", "predictive", "--variant", "world_model"]) == 0
    probe = json.loads(capsys.readouterr().out)["availability"]
    assert probe["variant"] == "world_model" and "python" in probe["backend_choices"]
    assert main(["codec", "inspect", "nonexistent"]) == 1
    assert "error" in json.loads(capsys.readouterr().err)


@pytest.mark.parametrize(("codec", "variant", "input_kind"), _VARIANTS)
def test_console_all_public_variants(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    codec: str,
    variant: str,
    input_kind: str,
) -> None:
    """Exercise partial tails, cross-read bytes and export every stated result type."""
    raster, waveform, rate = acquisition(None)
    original = raster if input_kind == "binary_raster" else waveform
    source, packet = tmp_path / "source.npy", tmp_path / "recording.sc-codec"
    decoded = tmp_path / "decoded.npz"
    np.save(source, original, allow_pickle=False)
    selection = [
        "--codec",
        codec,
        "--variant",
        variant,
        "--frame-rows",
        "257",
        "--sample-rate",
        str(rate),
        "--backend",
        "python",
    ]
    assert main(["codec", "encode", str(source), str(packet), *selection]) == 0
    encoded = json.loads(capsys.readouterr().out)
    assert encoded["file"]["variant"] == variant and encoded["backend"] == "python"
    native = "rust" in probe_codec(codec, variant).backend_choices
    decoder = "rust" if native else "python"
    if native:
        second = tmp_path / "native.sc-codec"
        assert main(["codec", "encode", str(source), str(second), *selection[:-1], decoder]) == 0
        assert json.loads(capsys.readouterr().out)["backend"] == "rust"
        assert second.read_bytes() == packet.read_bytes()
    assert main(["codec", "inspect", "--file", str(packet)]) == 0
    assert not json.loads(capsys.readouterr().out)["verified"]
    assert main(["codec", "verify", str(packet), "--backend", decoder]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["verified"] and status["backend"] == decoder
    assert main(["codec", "decode", str(packet), str(decoded), "--backend", decoder]) == 0
    fields = json.loads(capsys.readouterr().out)["fields"]
    with np.load(decoded, allow_pickle=False) as arrays:
        assert set(arrays.files) == {field["name"] for field in fields}
        for field in fields:
            values = arrays[field["name"]]
            assert values.shape == (field["rows"], field["channels"])
            assert values.dtype == np.dtype(field["dtype"])
        if input_kind == "binary_raster":
            # Timing precision defaults to unity, including on the lossy variant.
            np.testing.assert_array_equal(arrays["raster"], original)
        elif codec == "lossless_stream" or variant.startswith("lossless"):
            np.testing.assert_array_equal(arrays["waveform"], original)


def test_console_matched_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Report raw timings and stable packet digests from the same real input."""
    raster, _, _ = acquisition(None)
    source = tmp_path / "raster.npy"
    np.save(source, raster, allow_pickle=False)
    backend = "compare" if "rust" in probe_codec("isi").backend_choices else "python"
    command = ["codec", "benchmark", str(source), "--codec", "isi", "--repetitions", "2"]
    assert main([*command, "--backend", backend]) == 0
    report = json.loads(capsys.readouterr().out)
    runs = report["runs"]
    assert [run["backend"] for run in runs] == (
        ["python", "rust", "rust", "python"] if backend == "compare" else ["python", "python"]
    )
    assert len({run["packet_sha256"] for run in runs}) == 1
    assert len({run["decoded_sha256"] for run in runs}) == 1
    assert all(run["encode_ns"] > 0 and run["decode_ns"] > 0 for run in runs)
    assert main(["codec", "benchmark", str(source), "--codec", "isi", "--repetitions", "0"]) == 1
    assert "repetitions" in json.loads(capsys.readouterr().err)["error"]


@pytest.mark.parametrize(
    ("reshape_data", "reason"),
    [(True, "repetitions disagree"), (False, "input changed during measurement")],
)
def test_benchmark_refuses_input_changes_between_passes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    reshape_data: bool,
    reason: str,
) -> None:
    """Mutate the real input between real passes and require the benchmark to refuse it."""
    raster, _, _ = acquisition(None)
    source = tmp_path / "raster.npy"
    np.save(source, raster, allow_pickle=False)
    open_samples = samples.open_sample_input
    opened = 0

    @contextmanager
    def change_between_passes(
        path: Path,
        rate: int | None,
        *,
        event_rows: int | None,
        event_channels: int | None,
    ) -> Iterator[samples.SampleInput]:
        nonlocal opened
        opened += 1
        if opened == 2:
            if reshape_data:
                np.save(path, 1 - raster, allow_pickle=False)
            else:
                with path.open("ab") as stream:
                    stream.write(b"additional trailing recording metadata")
        with open_samples(
            path, rate, event_rows=event_rows, event_channels=event_channels
        ) as actual:
            yield actual

    monkeypatch.setattr(samples, "open_sample_input", change_between_passes)
    rc = main(
        [
            "codec",
            "benchmark",
            str(source),
            "--codec",
            "isi",
            "--backend",
            "python",
            "--repetitions",
            "2",
        ]
    )
    assert rc == 1
    status = capsys.readouterr()
    assert not status.out
    assert reason in json.loads(status.err)["error"]
