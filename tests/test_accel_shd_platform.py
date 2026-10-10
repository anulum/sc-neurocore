# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native non-Linux SHD refusal

"""Exercise the public reader's platform contract on the actual non-Linux runtime."""

from pathlib import Path
import sys
from typing import Literal

import pytest

from sc_neurocore.accel.shd_recordings import read_shd_recording


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="requires a real non-Linux runtime")
@pytest.mark.parametrize("backend", ["julia", "mojo"])
def test_public_reader_refuses_executable_launch_without_linux_parent_death_signals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: Literal["julia", "mojo"]
) -> None:
    """A configured real executable cannot bypass platform refusal before file decoding."""
    recording = tmp_path / "recording.h5"
    recording.touch()
    monkeypatch.setenv(f"SC_NEUROCORE_SHD_{backend.upper()}_EXE", sys.executable)
    with pytest.raises(RuntimeError, match="require Linux parent-death signals"):
        read_shd_recording(recording, 0, backend=backend)
