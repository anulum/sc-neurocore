# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Host driver input-guard tests

"""``generate_host_driver`` fails closed on malformed AXI-Lite/Wishbone driver inputs.

The generated driver is memory-mapped register access, so its boundary is the Q-format,
the base address and each parameter register's bit width — a malformed one would emit a
driver with degenerate or unbounded runtime masks. Every guard raises before any code is
generated.
"""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.host_driver_gen import generate_host_driver


def test_rejects_data_width_above_ceiling() -> None:
    with pytest.raises(ValueError, match="outside the supported range"):
        generate_host_driver("sc_lif", {"v": 16}, data_width=128, fraction=8)


def test_rejects_fraction_not_below_data_width() -> None:
    with pytest.raises(ValueError, match="0 <= fraction < data_width"):
        generate_host_driver("sc_lif", {"v": 16}, data_width=16, fraction=16)


def test_rejects_negative_base_address() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        generate_host_driver("sc_lif", {"v": 16}, base_address=-1)


def test_rejects_zero_parameter_width() -> None:
    with pytest.raises(ValueError, match="is outside the range"):
        generate_host_driver("sc_lif", {"v": 0})


def test_rejects_oversized_parameter_width() -> None:
    with pytest.raises(ValueError, match="is outside the range"):
        generate_host_driver("sc_lif", {"v": 100_000})


def test_valid_driver_still_generates() -> None:
    src = generate_host_driver("sc_lif", {"v": 16, "tau": 16}, data_width=16, fraction=8)
    assert len(src) > 100  # a real driver body, not an empty string
