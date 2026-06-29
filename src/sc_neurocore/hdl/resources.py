# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Access to baseline HDL primitives shipped inside SC-NeuroCore wheels."""

from __future__ import annotations

from importlib.resources import files
import sys

if sys.version_info >= (3, 11):
    from importlib.resources.abc import Traversable
else:  # pragma: no cover - import path exercised only on Python < 3.11
    from importlib.abc import Traversable

BASELINE_PRIMITIVE_RTL: tuple[str, ...] = (
    "sc_bitstream_encoder.v",
    "sc_bitstream_synapse.v",
    "sc_dense_layer_core.v",
    "sc_dotproduct_to_current.v",
    "sc_firing_rate_bank.v",
    "sc_lif_neuron.v",
)
_BASELINE_PRIMITIVE_SET = frozenset(BASELINE_PRIMITIVE_RTL)


def list_baseline_primitive_rtl() -> tuple[str, ...]:
    """Return the static baseline Verilog primitives bundled with the wheel."""

    return BASELINE_PRIMITIVE_RTL


def baseline_primitive_path(name: str) -> Traversable:
    """Return an importlib resource handle for a known baseline primitive."""

    if name not in _BASELINE_PRIMITIVE_SET:
        raise ValueError(f"Unknown baseline HDL primitive: {name!r}")
    resource = files("sc_neurocore.hdl.primitives").joinpath(name)
    if not resource.is_file():
        raise FileNotFoundError(f"Missing packaged HDL primitive: {name}")
    return resource


def baseline_primitive_text(name: str) -> str:
    """Read a known baseline primitive as UTF-8 text."""

    return baseline_primitive_path(name).read_text(encoding="utf-8")
