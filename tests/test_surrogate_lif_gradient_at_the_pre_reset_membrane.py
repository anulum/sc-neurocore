# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — surrogate LIF gradient at the pre-reset membrane tests

"""A surrogate LIF's gradient on a spike step is taken at the membrane that crossed threshold, before the reset.

Neftci, Mostafa and Zenke (2019, arXiv:1901.09948v2, p. 5) write the spike as ``S[n] = Θ(U[n] − ϑ)`` and apply the
reset to ``U[n + 1]`` (their eq. 5), so the surrogate derivative stands in for ``Θ'`` at the membrane before the
reset. The engine cached the voltage after the step, which is the reset voltage on a spike step: with the defaults a
spike got about 1/24 of the gradient this test's step should get. A refractory step returns no spike and the held
``v_rest`` whatever the membrane is, so its gradient is exactly 0, the derivative of the forward pass. The cases run
through the extension's class and through ``sc_neurocore_engine.grad.SurrogateLif``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.grad import SurrogateLif as WrapperSurrogateLif
from sc_neurocore_engine.sc_neurocore_engine import SurrogateLif as NativeSurrogateLif

_CLASSES = {"native": NativeSurrogateLif, "wrapper": WrapperSurrogateLif}


def _fast_sigmoid(x: float, k: float = 25.0) -> float:
    """Return the engine's fast-sigmoid derivative ``1 / (2k (1 + k|x|)^2)`` in single precision.

    Parameters
    ----------
    x : float
        The membrane's offset from threshold, in units of ``2^fraction``.
    k : float
        The steepness.
    """
    k32 = np.float32(k)
    denominator = np.float32(1.0) + k32 * np.float32(abs(x))
    return float(np.float32(1.0) / (np.float32(2.0) * k32 * denominator * denominator))


def _neuron(path: str, v_reset: int = 0) -> Any:
    """Return a default fast-sigmoid surrogate LIF (threshold 256, fraction 8) through ``path``.

    Parameters
    ----------
    path : str
        ``native`` or ``wrapper``.
    v_reset : int
        The reset voltage.
    """
    return _CLASSES[path](v_reset=v_reset, surrogate="fast_sigmoid", k=25.0)


@pytest.mark.parametrize("path", sorted(_CLASSES))
def test_a_spike_step_is_differentiated_at_the_membrane_that_crossed_threshold(path: str) -> None:
    """From 0 with no leak, a current of 300 integrates to 300, crosses 256 and resets to 0.

    Parameters
    ----------
    path : str
        ``native`` or ``wrapper``.
    """
    lif = _neuron(path)
    assert tuple(lif.forward(0, 256, 300)) == (1, 0)
    gradient = lif.backward(1.0)
    assert gradient == pytest.approx(_fast_sigmoid(44 / 256), rel=1e-6)
    assert gradient > 20 * _fast_sigmoid(-1.0)


@pytest.mark.parametrize("path", sorted(_CLASSES))
def test_a_refractory_steps_gradient_is_zero(path: str) -> None:
    """While refractory the output does not depend on the membrane, for the whole period.

    Parameters
    ----------
    path : str
        ``native`` or ``wrapper``.
    """
    lif = _neuron(path, v_reset=-128)
    assert tuple(lif.forward(0, 256, 300)) == (1, -128)
    assert tuple(lif.forward(0, 256, 300)) == (0, 0)
    assert tuple(lif.forward(0, 256, 300)) == (0, 0)
    assert tuple(lif.forward(0, 256, 100)) == (0, 100)
    assert lif.backward(1.0) == pytest.approx(_fast_sigmoid(-156 / 256), rel=1e-6)
    assert lif.backward(1.0) == 0.0
    assert lif.backward(1.0) == 0.0
    assert lif.backward(1.0) == pytest.approx(_fast_sigmoid(44 / 256), rel=1e-6)


@pytest.mark.parametrize("path", sorted(_CLASSES))
def test_a_step_below_threshold_is_differentiated_at_its_membrane(path: str) -> None:
    """Below threshold the membrane is the voltage the step returns, as before.

    Parameters
    ----------
    path : str
        ``native`` or ``wrapper``.
    """
    lif = _neuron(path)
    assert tuple(lif.forward(0, 256, 100)) == (0, 100)
    assert lif.backward(1.0) == pytest.approx(_fast_sigmoid(-156 / 256), rel=1e-6)
