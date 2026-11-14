# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The RTL compile refuses in words what it cannot bind

"""An equation the RTL cannot bind is refused in words, not emitted or failed in a fault.

Measured before this was written: ``dv/dt = -v / tau`` with no ``tau`` compiled to
RTL that read ``tau`` as a net declared nowhere, which Verilator refuses to
elaborate; an unknown function raised a plain ``ValueError``; and a reset of a name
that is not a state raised ``KeyError``. The Python model may still read a name
supplied at run time, so these are refusals of the RTL compile only.
"""

from typing import Any

import pytest

from sc_neurocore.compiler.equation_compiler import equation_to_fpga
from sc_neurocore.compiler.rtl_refusals import RtlCompileRefusal
from sc_neurocore.neurons.equation_builder import from_equations
from sc_neurocore.refusals import AuthoredRefusal

#: A valid one-ODE neuron; each refused case changes one field.
BASE: dict[str, Any] = {
    "threshold": "v > 0.9",
    "reset": "v = 0",
    "params": {},
    "init": {"v": 0.0},
}

#: Equations, the changed field, and the whole refusal the caller must be given.
REFUSED: list[tuple[tuple[str, ...], dict[str, Any], str]] = [
    (
        ("dv/dt = -v / tau",),
        {},
        "the equation reads 'tau', which is not a state, a parameter or a constant of "
        "this neuron; the RTL has no signal for it",
    ),
    (
        ("dv/dt = frobnicate(v)",),
        {},
        "Unsupported function 'frobnicate' in Verilog compilation.",
    ),
    (
        ("dv/dt = (-v + 1) / 10",),
        {"reset": "w = 0"},
        "the reset assigns 'w', which is not a state of this neuron",
    ),
]


@pytest.mark.parametrize(("equations", "change", "message"), REFUSED)
def test_the_rtl_compile_refuses_what_it_cannot_bind_in_words(
    equations: tuple[str, ...], change: dict[str, Any], message: str
) -> None:
    """Each input is refused by an authored RTL compile refusal with its whole text.

    Parameters
    ----------
    equations : tuple[str, ...]
        The ODE strings.
    change : dict[str, Any]
        The field that differs from the valid neuron.
    message : str
        The whole refusal.
    """
    with pytest.raises(RtlCompileRefusal) as refused:
        equation_to_fpga(*equations, **{**BASE, **change})
    assert str(refused.value) == message
    assert isinstance(refused.value, AuthoredRefusal)
    assert isinstance(refused.value, ValueError)


def test_a_name_given_at_run_time_still_drives_the_python_model() -> None:
    """The Python model reads a name passed to its step; only the RTL has no signal for it."""
    neuron = from_equations("dv/dt = -v + g", threshold="v > 0.9", reset="v = 0", init={"v": 0.0})
    neuron.step(I=0.0, g=1.0)
    assert neuron.state["v"] > 0.0


def test_every_name_the_rtl_binds_still_compiles() -> None:
    """A state, a parameter, the input and a named constant each have a signal or a literal."""
    _, verilog = equation_to_fpga(
        "dv/dt = -v / tau + I + 0.001 * pi",
        threshold="v > 0.9",
        reset="v = 0",
        params={"tau": 10.0},
        init={"v": 0.0},
        module_name="sc_bound_names",
    )
    assert "module sc_bound_names" in verilog
    assert "I_t" in verilog
