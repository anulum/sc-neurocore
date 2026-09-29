# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS runtime platform refusal

"""Exercise the public native reader's unsupported-platform admission decision."""

import sys
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.dvs_recordings import read_dvs_recording


def test_declared_unsupported_platform_refuses_before_native_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Controlled platform admission refuses an actual file/executable before launching it.

    This tests the refusal decision; it does not claim execution on another OS.
    """
    recording = tmp_path / "events.npy"
    np.save(recording, np.zeros((2, 4)))
    monkeypatch.setenv("SC_NEUROCORE_DVS_MOJO_EXE", sys.executable)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="requires Linux"):
        read_dvs_recording(recording, backend="mojo")
