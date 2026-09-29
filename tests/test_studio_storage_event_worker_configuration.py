# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Isolated event operator configuration refusal

"""Read real operator files and refuse invalid recording and runtime declarations."""

import json
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.storage_launcher_configuration import load_launcher_configuration
from tests.studio_storage_launcher_support import configuration


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_root": "relative"},
        {"dataset_root": "/operator/../dataset"},
        {"dataset_root": "/absent-event-worker-recordings"},
        {"input_max_bytes": 0},
        {"input_max_bytes": True},
        {"input_max_bytes": "64"},
        {"environment": {"PATH": "/arbitrary"}},
        {"go_library": "relative.so"},
        {"shd_go_library": "relative.so"},
        {"shd_rust_library": "relative.so"},
        {"shd_mojo_executable": "relative-mojo"},
        {"shd_mojo_executable": "/absent-shd-mojo"},
        {"shd_mojo_executable": "/"},
        {"shd_mojo_hdf5_library": "/lib/x86_64-linux-gnu/libc.so.6"},
        {"shd_julia_executable": "relative-julia"},
        {"shd_julia_executable": "/absent-shd-julia"},
        {"shd_julia_executable": "/"},
        {"shd_julia_hdf5_library": "/lib/x86_64-linux-gnu/libc.so.6"},
        {"shd_rust_library": "/absent-shd-rust-library.so"},
        {"shd_go_library": "/absent-shd-worker-library.so"},
        {"rust_library": "/absent-event-worker-library.so"},
        {"mojo_library": "/"},
        {"julia": {"executable": "/absent-julia", "project": "/", "handle_signals": "no"}},
        {"julia": {"executable": "julia", "project": "/", "handle_signals": "no"}},
        {"julia": {"executable": "/bin/sh", "project": "/", "handle_signals": "no"}},
        {"julia": {"executable": "/bin/sh", "project": "/", "handle_signals": "maybe"}},
    ],
)
def test_invalid_operator_event_settings_refuse_before_launcher_start(
    tmp_path: Path, change: dict[str, object]
) -> None:
    """Missing files, unsafe paths, invalid budgets and arbitrary environment refuse."""
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    path = tmp_path / "launcher.json"
    path.write_text(
        json.dumps(configuration(tmp_path, event_input={"dataset_root": str(recordings), **change}))
    )
    with pytest.raises(ValueError):
        load_launcher_configuration(path)


def test_directory_cannot_be_used_as_operator_julia_executable(tmp_path: Path) -> None:
    """A present directory is neither a Julia executable nor a library."""
    path = tmp_path / "launcher.json"
    path.write_text(
        json.dumps(
            configuration(
                tmp_path,
                event_input={
                    "dataset_root": str(tmp_path),
                    "julia": {
                        "executable": str(tmp_path),
                        "project": str(tmp_path),
                        "handle_signals": "no",
                    },
                },
            )
        )
    )
    with pytest.raises(ValueError, match="executable is unavailable"):
        load_launcher_configuration(path)
