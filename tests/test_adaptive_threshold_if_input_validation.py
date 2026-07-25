# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold accelerator input validation tests

"""Validate current-vector shape, finiteness, and allocation bounds."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.accel import adaptive_threshold_if as backends
from tests.adaptive_threshold_if_accel_dispatch_support import _PARAMETERS


def test_input_shape_and_finiteness_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        backends.simulate_adaptive_threshold_if(
            *_PARAMETERS,
            np.zeros((2, 2)),
            backend="python",
        )
    with pytest.raises(ValueError, match="finite"):
        backends.simulate_adaptive_threshold_if(
            *_PARAMETERS,
            [1.5, np.nan],
            backend="python",
        )


def test_signed_32_bit_step_bound_precedes_contiguous_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = np.broadcast_to(np.asarray([1.0]), ((1 << 31),))

    def unexpected_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("oversized logical input reached contiguous allocation")

    monkeypatch.setattr(np, "ascontiguousarray", unexpected_copy)
    with pytest.raises(ValueError, match="signed-32-bit step limit"):
        backends._input(oversized)
