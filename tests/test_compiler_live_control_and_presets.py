# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for live-control MMIO ops and precision presets

"""Contracts for live-control MMIO transaction validation and precision presets."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.compiler.live_control_ops import MMIORead, MMIOWrite
from sc_neurocore.compiler.live_control_types import MMIOReadPurpose, MMIOWritePurpose
from sc_neurocore.compiler.mixed_precision_spec import MixedPrecisionSpec
from sc_neurocore.compiler.precision_config import (
    BlockFloatingPrecisionConfig,
    PrecisionConfig,
)
from sc_neurocore.compiler.precision_presets import _parse_precision_spec, from_preset

_WRITE_PURPOSE = next(iter(MMIOWritePurpose))
_READ_PURPOSE = next(iter(MMIOReadPurpose))


def test_mmio_write_accepts_a_well_formed_transaction() -> None:
    """A 4-byte-aligned write whose value fits its width is accepted."""
    write = MMIOWrite(address_bytes=4, value=255, width_bits=8, purpose=_WRITE_PURPOSE)
    assert write.value == 255


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"address_bytes": True, "value": 1, "width_bits": 8}, "address_bytes must be an integer"),
        ({"address_bytes": 2, "value": 1, "width_bits": 8}, "4-byte aligned"),
        ({"address_bytes": 4, "value": True, "width_bits": 8}, "value must be an integer"),
        ({"address_bytes": 4, "value": 1, "width_bits": 7}, "width_bits must be one of"),
        ({"address_bytes": 4, "value": 256, "width_bits": 8}, "does not fit"),
    ],
)
def test_mmio_write_rejects_malformed_transactions(kwargs: dict[str, Any], message: str) -> None:
    """Each MMIO write invariant rejects the corresponding malformed field."""
    with pytest.raises(ValueError, match=message):
        MMIOWrite(purpose=_WRITE_PURPOSE, **kwargs)


def test_mmio_read_accepts_a_well_formed_transaction() -> None:
    """A 4-byte-aligned read with a supported width is accepted."""
    read = MMIORead(address_bytes=0, width_bits=32, purpose=_READ_PURPOSE)
    assert read.width_bits == 32


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"address_bytes": True, "width_bits": 8}, "address_bytes must be an integer"),
        ({"address_bytes": 2, "width_bits": 8}, "4-byte aligned"),
        ({"address_bytes": 4, "width_bits": 7}, "width_bits must be one of"),
    ],
)
def test_mmio_read_rejects_malformed_transactions(kwargs: dict[str, Any], message: str) -> None:
    """Each MMIO read invariant rejects the corresponding malformed field."""
    with pytest.raises(ValueError, match=message):
        MMIORead(purpose=_READ_PURPOSE, **kwargs)


def test_parse_precision_spec_passes_explicit_configs_through() -> None:
    """An already-resolved precision config is returned unchanged."""
    fixed = PrecisionConfig(8, 7)
    block = BlockFloatingPrecisionConfig(16, 3, 32)
    assert _parse_precision_spec(fixed) is fixed
    assert _parse_precision_spec(block) is block


def test_from_preset_resolves_named_format_and_explicit_specs() -> None:
    """from_preset resolves named presets, Q/BFP format strings and explicit configs."""
    explicit = PrecisionConfig(8, 7)
    spec = from_preset({"weights": "q88", "logits": "Q8.8", "acts": "bfp16e3x32", "bias": explicit})
    assert isinstance(spec, MixedPrecisionSpec)


def test_from_preset_rejects_unknown_named_preset() -> None:
    """An unrecognised preset name reports the available presets."""
    with pytest.raises(KeyError, match="Unknown preset"):
        from_preset({"weights": "totally_unknown"})


def test_from_preset_rejects_unsupported_preset_type() -> None:
    """A preset that is neither a string nor a config object is rejected."""
    with pytest.raises(TypeError):
        from_preset({"weights": 123})  # type: ignore[dict-item]


def test_crc32_update_guard_is_deterministic() -> None:
    """The CRC32 update guard is a deterministic 32-bit function of its four words."""
    from sc_neurocore.compiler.live_control_ops import _crc32_update_guard

    crc = _crc32_update_guard(1, 2, 3, 4)
    assert crc == _crc32_update_guard(1, 2, 3, 4)
    assert crc != _crc32_update_guard(1, 2, 3, 5)
    assert 0 <= crc <= 0xFFFF_FFFF
