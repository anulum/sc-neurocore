# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed Mojo DVS distribution acceptance

"""Run the public Mojo DVS reader from an actual extracted wheel away from the checkout."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np

from tests.test_accel_mojo_dvs import mojo_dvs_executable as mojo_dvs_executable
from tests.test_accel_julia_dvs_distribution import dvs_wheel as dvs_wheel
from tests.test_studio_distribution import distribution_source as distribution_source


def test_installed_wheel_mojo_reader_preserves_owned_values(
    dvs_wheel: Path, mojo_dvs_executable: Path, tmp_path: Path
) -> None:
    """The wheel-owned API launches its operator-owned compiled command and reproduces NumPy values."""
    recording = tmp_path / "events.npy"
    values = np.array([[1, 2, 1, 0.125], [3, 4, 0, 2.5]], dtype=">f8", order="F")
    np.save(recording, values)
    environment = dict(os.environ)
    environment["SC_NEUROCORE_DVS_MOJO_EXE"] = str(mojo_dvs_executable)
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
actual = dvs_recordings.read_dvs_recording(Path(sys.argv[2]), backend="mojo")
expected = np.load(sys.argv[2], allow_pickle=False)
np.testing.assert_array_equal(actual, expected)
assert actual.flags.owndata and actual.flags.writeable and actual.flags.c_contiguous
print("wheel Mojo DVS parity")
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
    assert result.stdout == "wheel Mojo DVS parity\n"
