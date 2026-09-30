# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed Julia DVS distribution acceptance

"""Run the public Julia DVS reader from an actual extracted wheel away from the checkout."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

import numpy as np
import pytest

from tests.test_accel_julia_dvs_backends import julia_dvs_executable as julia_dvs_executable
from tests.test_studio_distribution import build_distribution
from tests.test_studio_distribution import distribution_source as distribution_source


@pytest.fixture(scope="module", params=[False, True], ids=["wheel", "sdist-wheel"])
def dvs_wheel(
    distribution_source: Path,
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> Path:
    """Build direct and source-distribution wheels from current sources and native scripts."""
    root = Path(__file__).resolve().parents[1]
    # Tracked sources only, as an sdist carries: a walk of src/ also copied
    # ignored virtual environments (accel/go/.venv holds a whole Go toolchain
    # whose read-only .py files made the second build's copy fail).
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "src/*.py", "src/sc_neurocore/accel/julia/datasets/*.jl"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    inputs = [root / name for name in tracked.split("\0") if name]
    for source in inputs:
        destination = distribution_source / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    output = tmp_path_factory.mktemp("julia-dvs-wheel")
    build_source = distribution_source
    if request.param:
        built = build_distribution(build_source, "build_sdist", output / "sdist")
        assert built.returncode == 0, built.stdout + built.stderr
        (archive_path,) = (output / "sdist").glob("*.tar.gz")
        with tarfile.open(archive_path) as archive:
            archive.extractall(output / "unpacked", filter="data")
        (build_source,) = (output / "unpacked").iterdir()
    built = build_distribution(build_source, "build_wheel", output)
    assert built.returncode == 0, built.stdout + built.stderr
    (wheel,) = output.glob("*.whl")
    extracted = output / "extracted"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extracted)
    return extracted


def test_installed_wheel_julia_reader_preserves_owned_values(
    dvs_wheel: Path, julia_dvs_executable: Path, tmp_path: Path
) -> None:
    """The wheel-owned API launches its wheel-owned script and reproduces NumPy values."""
    recording = tmp_path / "events.npy"
    values = np.array([[1, 2, 1, 0.125], [3, 4, 0, 2.5]], dtype=">f8", order="F")
    np.save(recording, values)
    environment = dict(os.environ)
    environment["SC_NEUROCORE_DVS_JULIA_EXE"] = str(julia_dvs_executable)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from sc_neurocore.accel import dvs_recordings
assert Path(dvs_recordings.__file__).is_relative_to(Path(sys.argv[1]))
actual = dvs_recordings.read_dvs_recording(Path(sys.argv[2]), backend="julia")
expected = np.load(sys.argv[2], allow_pickle=False)
np.testing.assert_array_equal(actual, expected)
assert actual.flags.owndata and actual.flags.writeable and actual.flags.c_contiguous
print("wheel Julia DVS parity")
""",
            str(dvs_wheel),
            str(recording),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "wheel Julia DVS parity\n"
