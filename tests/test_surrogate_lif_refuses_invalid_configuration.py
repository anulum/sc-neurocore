# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — surrogate LIF configuration refusal tests

"""A refused surrogate LIF configuration raises ``ValueError``; it does not end the interpreter.

The engine's release build aborts on a panic, and the surrogate LIF's constructor panicked on
what the fixed-point LIF refuses, so ``SurrogateLif(data_width=17)`` ended the process with no
exception. Each case runs in a child interpreter, so a regression shows as a dead child rather
than a dead test run, through the native class and through the public wrapper.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_PATHS = {
    "native": "from sc_neurocore_engine.sc_neurocore_engine import SurrogateLif",
    "wrapper": "from sc_neurocore_engine.grad import SurrogateLif",
}

_REFUSED = {
    "data_width above 16": ("data_width=17", "data_width must be in [1, 16]"),
    "data_width 0": ("data_width=0, fraction=0", "data_width must be in [1, 16]"),
    "fraction not below data_width": (
        "data_width=16, fraction=16",
        "fraction must be less than data_width",
    ),
    "negative refractory period": ("refractory_period=-1", "refractory_period must be nonnegative"),
}


@pytest.mark.parametrize("path", sorted(_PATHS))
@pytest.mark.parametrize("case", sorted(_REFUSED))
def test_a_refused_configuration_raises_value_error_in_a_live_interpreter(
    path: str, case: str
) -> None:
    """The constructor raises ``ValueError`` with the fixed-point LIF's reason, and the child exits 0.

    Parameters
    ----------
    path : str
        ``native`` for the extension's class, ``wrapper`` for ``sc_neurocore_engine.grad``.
    case : str
        The refused configuration.
    """
    arguments, reason = _REFUSED[case]
    program = (
        f"{_PATHS[path]}\n"
        "try:\n"
        f"    SurrogateLif({arguments})\n"
        "except ValueError as error:\n"
        "    print('ValueError:', error)\n"
        "else:\n"
        "    print('accepted')\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120, check=False
    )
    assert child.returncode == 0, child.stderr[-2000:]
    assert child.stdout.strip().splitlines()[-1] == f"ValueError: {reason}"


@pytest.mark.parametrize("path", sorted(_PATHS))
def test_the_default_configuration_is_still_accepted(path: str) -> None:
    """The defaults construct a neuron through both paths.

    Parameters
    ----------
    path : str
        ``native`` for the extension's class, ``wrapper`` for ``sc_neurocore_engine.grad``.
    """
    program = f"{_PATHS[path]}\nlif = SurrogateLif()\nprint(lif.trace_len())\n"
    child = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=120, check=False
    )
    assert child.returncode == 0, child.stderr[-2000:]
    assert child.stdout.strip().splitlines()[-1] == "0"
