# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLayerPrecision from former test_adaptive_precision.py

"""Focused suite: TestLayerPrecision from former test_adaptive_precision.py."""

from __future__ import annotations

from tests.adaptive_precision_support import *  # noqa: F403


class TestLayerPrecision:
    """LayerPrecision data-contract checks."""

    def test_dataclass_fields(self) -> None:
        """LayerPrecision preserves assigned field values."""
        lp = LayerPrecision(
            layer_index=0,
            name="fc1",
            bitstream_length=256,
            error_bound=0.031,
            sensitivity=0.05,
        )
        assert lp.layer_index == 0
        assert lp.bitstream_length == 256

    def test_to_dict_serializes_manifest_row(self) -> None:
        """LayerPrecision exposes a deterministic manifest row."""
        lp = LayerPrecision(
            layer_index=1,
            name="classifier",
            bitstream_length=512,
            error_bound=0.015625,
            sensitivity=0.25,
        )

        assert lp.to_dict() == {
            "layer_index": 1,
            "name": "classifier",
            "bitstream_length": 512,
            "error_bound": 0.015625,
            "sensitivity": 0.25,
        }

    @pytest.mark.parametrize(
        ("factory", "message"),
        [
            (
                lambda: LayerPrecision(-1, "fc1", 256, 0.031, 0.05),
                "layer_index",
            ),
            (
                lambda: LayerPrecision(0, "", 256, 0.031, 0.05),
                "name",
            ),
            (
                lambda: LayerPrecision(0, "fc1", 0, 0.031, 0.05),
                "bitstream_length",
            ),
            (
                lambda: LayerPrecision(0, "fc1", 300, 0.031, 0.05),
                "power of two",
            ),
            (
                lambda: LayerPrecision(0, "fc1", 256, -0.1, 0.05),
                "error_bound",
            ),
            (
                lambda: LayerPrecision(0, "fc1", 256, 0.031, -0.1),
                "sensitivity",
            ),
        ],
    )
    def test_rejects_invalid_layer_precision_fields(
        self,
        factory: Callable[[], LayerPrecision],
        message: str,
    ) -> None:
        """LayerPrecision rejects impossible adaptive-length rows."""
        with pytest.raises(ValueError, match=message):
            factory()

    def test_rejects_non_numeric_layer_error_bound(self) -> None:
        """LayerPrecision rejects non-numeric error-bound payloads at runtime."""
        with pytest.raises(ValueError, match="error_bound"):
            LayerPrecision(
                layer_index=0,
                name="fc1",
                bitstream_length=256,
                error_bound=cast(float, "bad"),
                sensitivity=0.05,
            )
