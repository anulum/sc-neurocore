# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR conv string-padding ('same' / 'valid')

"""String padding modes ('same', 'valid') for NIR Conv1d/Conv2d import."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nir")

import nir

from sc_neurocore.nir_bridge.node_map import SCConv1dNode, SCConv2dNode


def _conv1d(padding, *, kernel=3, dilation=1, stride=1) -> nir.Conv1d:
    return nir.Conv1d(
        input_shape=8,
        weight=np.ones((2, 1, kernel)),
        bias=np.zeros(2),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
    )


def _conv2d(padding, *, kernel=(3, 3), dilation=(1, 1), stride=(1, 1)) -> nir.Conv2d:
    return nir.Conv2d(
        input_shape=(8, 8),
        weight=np.ones((2, 1, kernel[0], kernel[1])),
        bias=np.zeros(2),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1,
    )


def test_conv1d_valid_padding_is_zero() -> None:
    node = SCConv1dNode.from_nir("c", _conv1d("valid"))
    assert node.padding == 0


def test_conv1d_same_padding_preserves_length() -> None:
    node = SCConv1dNode.from_nir("c", _conv1d("same", kernel=3))
    assert node.padding == 1
    out = node.forward(np.ones((1, 8)))
    assert out.shape[-1] == 8  # length preserved


def test_conv1d_same_with_dilation_preserves_length() -> None:
    node = SCConv1dNode.from_nir("c", _conv1d("same", kernel=3, dilation=2))
    assert node.padding == 2  # dilation * (k - 1) // 2 = 2 * 2 // 2
    out = node.forward(np.ones((1, 10)))
    assert out.shape[-1] == 10


def test_conv1d_same_rejects_even_kernel() -> None:
    with pytest.raises(ValueError, match="even effective kernel span"):
        SCConv1dNode.from_nir("c", _conv1d("same", kernel=4))


def test_conv1d_same_rejects_stride_above_one() -> None:
    with pytest.raises(ValueError, match="requires stride 1"):
        SCConv1dNode.from_nir("c", _conv1d("same", stride=2))


def test_resolve_conv_padding_rejects_unknown_mode() -> None:
    # nir itself rejects non-same/valid strings at construction, so exercise the
    # resolver's defensive branch directly.
    from sc_neurocore.nir_bridge.node_map import _resolve_conv_padding

    with pytest.raises(ValueError, match="Unsupported conv padding mode"):
        _resolve_conv_padding("reflect", kernel=3, dilation=1, stride=1)


def test_conv1d_integer_padding_still_passes_through() -> None:
    node = SCConv1dNode.from_nir("c", _conv1d(2))
    assert node.padding == 2


def test_conv2d_same_padding_preserves_spatial_size() -> None:
    node = SCConv2dNode.from_nir("c", _conv2d("same", kernel=(3, 3)))
    assert node.padding == (1, 1)
    out = node.forward(np.ones((1, 8, 8)))
    assert out.shape[-2:] == (8, 8)


def test_conv2d_valid_padding_is_zero() -> None:
    node = SCConv2dNode.from_nir("c", _conv2d("valid"))
    assert node.padding == (0, 0)


def test_conv2d_integer_tuple_padding_passes_through() -> None:
    node = SCConv2dNode.from_nir("c", _conv2d((1, 2)))
    assert node.padding == (1, 2)
