# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia recording decoding in actual training children

"""Exercise JuliaCall in the same public saved-state acceptance as compiled readers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.test_studio_event_training_native import (
    test_native_child_matches_numpy_saved_state_and_refuses_missing_library as _assert_child_parity,
)


def test_julia_child_matches_numpy_saved_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual HTTP jobs decode with Julia, retain exact state and refuse invalid runtime."""
    project = Path(os.environ["PYTHON_JULIACALL_PROJECT"])
    _assert_child_parity(tmp_path, ("julia", project), monkeypatch)
