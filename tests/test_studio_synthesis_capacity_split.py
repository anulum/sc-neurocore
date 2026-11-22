# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio synthesis and capacity judgement file division contract

"""The capacity judgement left the toolchain module; every historical import still reaches it.

``sc_neurocore.studio.synthesis`` held two responsibilities: running the bounded
FPGA toolchains, and judging whether a synthesised design fits its target
device. The judgement now lives in ``sc_neurocore.studio.synthesis_capacity``.
The toolchain module re-exports its public names as the same objects, so code
and pickles that name ``sc_neurocore.studio.synthesis.capacity_verdict`` keep
working, and the toolchain still records the verdict the new module returns.
"""

from __future__ import annotations

import pickle

import sc_neurocore.studio.synthesis as synthesis
import sc_neurocore.studio.synthesis_capacity as synthesis_capacity

PUBLIC_CAPACITY_NAMES = ("capacity_device_name", "capacity_sentence", "capacity_verdict")


def test_historical_synthesis_imports_are_the_capacity_module_objects() -> None:
    """Each public capacity name of the toolchain module is the capacity module's own object."""
    for name in PUBLIC_CAPACITY_NAMES:
        assert getattr(synthesis, name) is getattr(synthesis_capacity, name), name
    assert set(PUBLIC_CAPACITY_NAMES) == set(synthesis_capacity.__all__)


def test_the_judgement_is_defined_in_its_own_module() -> None:
    """The judging functions are owned by the capacity module, by name and by pickle."""
    for name in PUBLIC_CAPACITY_NAMES:
        function = getattr(synthesis_capacity, name)
        assert function.__module__ == "sc_neurocore.studio.synthesis_capacity", name
        assert function.__qualname__ == name
        assert pickle.loads(pickle.dumps(getattr(synthesis, name))) is function


def test_the_toolchain_module_no_longer_carries_the_judged_resources() -> None:
    """The judged resource keys moved with the verdict; the estimate still records a verdict."""
    assert "_CAPACITY_KEYS" not in vars(synthesis)
    assert vars(synthesis_capacity)["_CAPACITY_KEYS"] == ("luts", "ffs", "brams", "dsps")
    estimate = synthesis.estimate_resources(4, target="ice40")
    assert estimate["capacity_device"] == synthesis_capacity.capacity_device_name("ice40")
    assert estimate["fits_device"] is True
