# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Named Unicode escapes in native DVS metadata

"""Compare actual native recording APIs for named underscores in valid keys and scalar aliases."""

from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording
from tests.test_accel_go_dvs_recordings import go_dvs_executable as go_dvs_executable
from tests.test_accel_julia_dvs_backends import julia_dvs_executable as julia_dvs_executable
from tests.test_accel_julia_dvs_literals import recording
from tests.test_accel_rust_dvs import rust_dvs_executable as rust_dvs_executable


@pytest.mark.parametrize("backend", ["go", "rust", "julia"])
def test_named_underscore_preserves_scalar_alias_and_field_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    go_dvs_executable: Path,
    rust_dvs_executable: Path,
    julia_dvs_executable: Path,
    backend: Literal["go", "rust", "julia"],
) -> None:
    """Valid Unicode LOW LINE names retain keys and bool_/int_ values without fallback."""
    executable = {
        "go": go_dvs_executable,
        "rust": rust_dvs_executable,
        "julia": julia_dvs_executable,
    }[backend]
    monkeypatch.setenv(f"SC_NEUROCORE_DVS_{backend.upper()}_EXE", str(executable))
    for index, (descriptor, name) in enumerate((("bool", "LOW LINE"), ("int", "low line"))):
        values = (np.arange(8).reshape((2, 4)) % 2).astype(descriptor + "_")
        header = (
            "{'descr':'" + descriptor + "\\N{" + name + "}',"
            "'fortran\\N{" + name + "}order':False,'shape':(2,4)}\n"
        )
        path = recording(tmp_path, index, header, values.tobytes())
        expected = read_dvs_recording(path, backend="numpy")
        actual = read_dvs_recording(path, backend=backend)
        np.testing.assert_array_equal(actual, expected)
        assert actual.flags.owndata and actual.flags.writeable and actual.flags.c_contiguous
