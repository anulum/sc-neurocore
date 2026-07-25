# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS input bitstream contracts

"""Focused DVS input bitstream contracts."""

from tests.interfaces.dvs_input_support import *


def test_dvs_generate_bitstream_frame_shape() -> None:
    """Bitstream frame should be (H, W, length)."""
    layer = DVSInputLayer(height=2, width=3)
    bits = layer.generate_bitstream_frame(length=8)
    assert bits.shape == (2, 3, 8)


def test_dvs_generate_bitstream_frame_binary() -> None:
    """Bitstream frame should be binary."""
    layer = DVSInputLayer(height=2, width=2)
    bits = layer.generate_bitstream_frame(length=4)
    assert set(np.unique(bits).tolist()) <= {0, 1}


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_dvs_rejects_invalid_bitstream_length(length: Any) -> None:
    """Generated bitstream frames require a positive integer sample length."""
    layer = DVSInputLayer(height=2, width=2)
    with pytest.raises(ValueError, match="length must be a positive integer"):
        layer.generate_bitstream_frame(length=length)
