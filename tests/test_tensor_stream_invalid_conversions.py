# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestInvalidConversions from former test_tensor_stream.py

"""Focused suite: TestInvalidConversions from former test_tensor_stream.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.core.tensor_stream import (
    TensorStream,
)


class TestInvalidConversions:
    """Conversions that are not defined."""

    def test_spike_to_bitstream_raises(self) -> None:
        """A spike tensor cannot become a bitstream."""
        ts = TensorStream(data=np.array([1]), domain="spike")
        with pytest.raises(ValueError):
            ts.to_bitstream()
