# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go indexed HDF5 reader parity

"""Read actual HDF5 rows through the public Go API and compare the Python reader."""

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from sc_neurocore.datasets.event_samples import read_event_sample
from sc_neurocore.datasets.manifest import SampleRecord


@pytest.fixture(scope="module")
def go_shd_reader(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a real caller of the native indexed reader with explicit HDF5 opt-in."""
    directory = tmp_path_factory.mktemp("go-shd-reader")
    main = directory / "main.go"
    main.write_text(
        "\n".join(
            (
                Path(__file__).resolve().parents[1]
                / "src/sc_neurocore/accel/go/services/loaders/shd.go"
            )
            .read_text()
            .splitlines()[:7]
        )
        + "\n\n"
        + """package main
import("encoding/json";"fmt";"os";"strconv";
"github.com/anulum/sc-neurocore/accel/services/loaders")
func main(){
 index,err:=strconv.Atoi(os.Args[2]);if err!=nil{panic(err)}
 budget,err:=strconv.Atoi(os.Args[3]);if err!=nil{panic(err)}
 recording,err:=loaders.ReadSHDRecording(os.Args[1],index,budget)
 if err!=nil{fmt.Fprintln(os.Stderr,err);os.Exit(1)}
 if err=json.NewEncoder(os.Stdout).Encode(recording);err!=nil{panic(err)}
}
"""
    )
    executable = directory / "read-shd"
    subprocess.run(
        ["go", "build", "-tags=hdf5", "-o", str(executable), str(main)],
        cwd=Path(__file__).resolve().parents[1] / "src/sc_neurocore/accel/go",
        check=True,
        capture_output=True,
        timeout=120,
    )
    return executable


@pytest.fixture
def shd_file(tmp_path: Path) -> Path:
    """Store finite float16 spike times, integer channels and an empty second row."""
    path = tmp_path / "recordings.h5"
    with h5py.File(path, "w") as handle:
        times = handle.create_dataset("spikes/times", (2,), dtype=h5py.vlen_dtype(np.float16))
        units = handle.create_dataset("spikes/units", (2,), dtype=h5py.vlen_dtype(np.int16))
        times[0] = np.array([0, 0.000333, 0.9995], dtype=np.float16)
        units[0] = np.array([0, 34, 699], dtype=np.int16)
        times[1] = np.array([], dtype=np.float16)
        units[1] = np.array([], dtype=np.int16)
        handle["labels"] = np.array([3, 17], dtype=np.uint8)
    return path


@pytest.mark.parametrize("index,label", [(0, 3), (1, 17)])
def test_go_indexed_shd_reader_preserves_widened_times_and_empty_rows(
    go_shd_reader: Path, shd_file: Path, index: int, label: int
) -> None:
    """Native HDF5 conversion agrees exactly with the manifest sample reader."""
    result = subprocess.run(
        [str(go_shd_reader), str(shd_file), str(index), "1024"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    record = json.loads(result.stdout)
    sample = SampleRecord(
        split="train", file=shd_file.name, index=index, label=label, group="speaker"
    )
    expected = read_event_sample(shd_file.parent, "shd", sample)
    np.testing.assert_array_equal(np.asarray(record["Events"]).reshape(-1, 4), expected)
    assert record["Label"] == label


@pytest.mark.parametrize("index,budget", [(-1, 1024), (2, 1024), (0, -1), (0, 95)])
def test_go_indexed_shd_reader_refuses_invalid_index_or_event_budget(
    go_shd_reader: Path, shd_file: Path, index: int, budget: int
) -> None:
    """Invalid inputs fail before returning a partial or substituted recording."""
    result = subprocess.run(
        [str(go_shd_reader), str(shd_file), str(index), str(budget)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert result.stdout == ""


@pytest.mark.parametrize("damage", ["missing", "shape", "pair", "label"])
def test_go_indexed_shd_reader_refuses_incompatible_hdf5_rows(
    go_shd_reader: Path, shd_file: Path, damage: str
) -> None:
    """Missing files, wrong row counts, unmatched vectors and noninteger labels refuse."""
    if damage == "missing":
        shd_file = shd_file.with_name("absent.h5")
    else:
        with h5py.File(shd_file, "a") as handle:
            if damage == "shape":
                del handle["labels"]
                handle["labels"] = np.array([3], dtype=np.uint8)
            elif damage == "pair":
                handle["spikes/units"][0] = np.array([1], dtype=np.int16)
            else:
                del handle["labels"]
                handle["labels"] = np.array([3, 17], dtype=np.float64)
    result = subprocess.run(
        [str(go_shd_reader), str(shd_file), "0", "1024"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert result.stdout == ""
