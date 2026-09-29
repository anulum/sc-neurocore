# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed event operator configuration acceptance

"""Read operator JSON with actual compiled decoders and installed standalone Julia."""

import json
import os
from pathlib import Path
from typing import Literal

import pytest

from sc_neurocore.studio.platform.storage_launcher_configuration import load_launcher_configuration
from tests.studio_storage_launcher_support import configuration
from tests.test_accel_event_recordings import (
    go_recording_library as go_recording_library,
    mojo_recording_library as mojo_recording_library,
    native_recording_library as native_recording_library,
    rust_recording_library as rust_recording_library,
)
from tests.test_accel_julia_shd import julia_shd_executable as julia_shd_executable


@pytest.mark.parametrize("explicit_hdf5", [False, True])
def test_operator_json_preserves_native_nmnist_and_independent_julia_shd(
    tmp_path: Path,
    native_recording_library: tuple[Literal["go", "rust", "mojo"], Path],
    julia_shd_executable: Path,
    explicit_hdf5: bool,
) -> None:
    """Real decoder artifacts remain distinct across the N-MNIST and SHD selections."""
    backend, library = native_recording_library
    hdf5 = Path("/lib/x86_64-linux-gnu/libhdf5_serial.so").resolve(strict=True)
    operator = {
        "dataset_root": str(tmp_path),
        f"{backend}_library": str(library),
        "shd_julia_executable": str(julia_shd_executable),
    }
    if explicit_hdf5:
        operator["shd_julia_hdf5_library"] = str(hdf5)
    path = tmp_path / "launcher.json"
    path.write_text(json.dumps(configuration(tmp_path, event_input=operator)))
    loaded = load_launcher_configuration(path)
    assert loaded.event_input is not None
    environment = loaded.event_input.environment()
    assert environment[f"SC_NEUROCORE_DATASET_{backend.upper()}_LIBRARY"] == str(library)
    assert environment["SC_NEUROCORE_SHD_JULIA_EXE"] == str(julia_shd_executable)
    if explicit_hdf5:
        assert environment["SC_NEUROCORE_SHD_JULIA_HDF5_LIBRARY"] == str(hdf5)
    else:
        assert "SC_NEUROCORE_SHD_JULIA_HDF5_LIBRARY" not in environment
    assert "SC_NEUROCORE_DATASET_JULIA_ENABLED" not in environment
    assert not any(key.startswith("SC_NEUROCORE_DVS_") for key in environment)


def test_operator_json_preserves_installed_juliacall_runtime_policy(
    tmp_path: Path, julia_shd_executable: Path
) -> None:
    """An explicitly provisioned project retains its CPU-thread and signal declarations."""
    project = Path(os.environ["PYTHON_JULIACALL_PROJECT"]).resolve(strict=True)
    operator = {
        "dataset_root": str(tmp_path),
        "julia": {
            "executable": str(julia_shd_executable),
            "project": str(project),
            "handle_signals": "no",
        },
    }
    path = tmp_path / "launcher.json"
    path.write_text(json.dumps(configuration(tmp_path, event_input=operator)))
    loaded = load_launcher_configuration(path)
    assert loaded.event_input is not None
    environment = loaded.event_input.environment()
    assert environment["SC_NEUROCORE_DATASET_JULIA_ENABLED"] == "1"
    assert environment["PYTHON_JULIACALL_EXE"] == str(julia_shd_executable)
    assert environment["PYTHON_JULIACALL_PROJECT"] == str(project)
    assert environment["PYTHON_JULIACALL_THREADS"] == "1"
    assert environment["PYTHON_JULIACALL_HANDLE_SIGNALS"] == "no"
    assert "SC_NEUROCORE_SHD_JULIA_EXE" not in environment
