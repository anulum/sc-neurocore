# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic network benchmark CLI repetition records

"""Exercise complete NumPy and native benchmark records through the real CLI."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import TypedDict, cast

import pytest

from tests.engine_requirement import require_engine

require_engine()


class _Run(TypedDict):
    """Describe one measured simulation repetition in the public report."""

    n_neurons: int
    bitstream_length: int
    backend: str
    repetition: int
    seed: int
    sim_steps: int
    encode_s: float
    mac_s: float
    decode_s: float
    total_s: float
    encode_gbit_s: float
    mac_gop_s: float
    total_spikes: int
    firing_rate: float


class _Report(TypedDict):
    """Describe the public report's protocol, timestamp and measured rows."""

    schema_version: int
    timestamp: str
    protocol: dict[str, object]
    component_timing: dict[str, str]
    data: list[_Run]


@pytest.mark.parametrize(
    "scales,bitstream_lengths,steps,repeats",
    [([100], [256], 50, 3), ([4, 8], [64, 128], 3, 2)],
)
def test_cli_retains_every_numpy_and_native_repetition(
    tmp_path: Path, scales: list[int], bitstream_lengths: list[int], steps: int, repeats: int
) -> None:
    """Preserve all original repetitions in JSON and Markdown without last-run loss.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Isolated working directory and destination for the actual CLI output.
    scales : list of int
        Every neuron count requested from both real backends.
    bitstream_lengths : list of int
        Every bitstream length requested for each neuron count.
    steps : int
        Complete steps per repetition.
    repeats : int
        Repetitions retained per configuration and backend.
    """
    script = Path(__file__).resolve().parents[1] / "benchmarks/sc_network_benchmark.py"
    output = tmp_path / "measurements.json"
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    before = datetime.now(timezone.utc)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(script),
            "--scales",
            *map(str, scales),
            "--bitstream-lengths",
            *map(str, bitstream_lengths),
            "--sim-steps",
            str(steps),
            "--repeats",
            str(repeats),
            "--json",
            str(output),
            "--markdown",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    after = datetime.now(timezone.utc)
    report = cast(_Report, json.loads(output.read_text(encoding="utf-8")))
    assert len(report["data"]) == len(scales) * len(bitstream_lengths) * 2 * repeats
    assert {row["backend"] for row in report["data"]} == {"numpy", "rust"}
    assert report["schema_version"] == 2
    assert report["protocol"] == {
        "scales": scales,
        "bitstream_lengths": bitstream_lengths,
        "sim_steps": steps,
        "repeats": repeats,
        "initial_seed": 42,
    }
    timestamp = datetime.fromisoformat(report["timestamp"])
    assert timestamp.tzinfo == timezone.utc
    assert before <= timestamp <= after
    assert report["component_timing"]["rust"] == (
        "measured fused total; estimated 40/50/10 stage split"
    )
    groups = [
        (n, bl, backend)
        for n in scales
        for bl in bitstream_lengths
        for backend in ("numpy", "rust")
    ]
    for n, bl, backend in groups:
        rows = [
            row
            for row in report["data"]
            if (row["n_neurons"], row["bitstream_length"], row["backend"]) == (n, bl, backend)
        ]
        assert [(row["repetition"], row["seed"], row["sim_steps"]) for row in rows] == [
            (rep, 42 + rep, steps) for rep in range(repeats)
        ]
        for row in rows:
            assert row["total_s"] > 0
            assert math.isclose(row["total_s"], row["encode_s"] + row["mac_s"] + row["decode_s"])
            assert 0 <= row["total_spikes"] <= n * steps
            assert row["firing_rate"] == row["total_spikes"] / (n * steps)
        mean = re.search(
            rf"{backend} n=\s*{n} bl=\s*{bl}\.\.\. mean_total=([0-9.]+)s", result.stdout
        )
        assert mean is not None
        assert abs(float(mean.group(1)) - statistics.mean(row["total_s"] for row in rows)) <= (
            0.000500001
        )
    table_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in result.stdout.splitlines()
        if re.match(r"^\|\s+(numpy|rust)\s*\|", line)
    ]
    assert [(row[0], int(row[3]), int(row[4])) for row in table_rows] == [
        (row["backend"], row["repetition"], row["seed"]) for row in report["data"]
    ]
    assert "Rust component rates use an estimated 40/50/10 stage split" in result.stdout
