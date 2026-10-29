# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Held equation state lowered to explicit state transitions

"""Preserve authored state without a differential or map equation in compiled targets."""

from __future__ import annotations

from copy import copy

from sc_neurocore.neurons.equation_builder import EquationNeuron


def lower_held_state(neuron: EquationNeuron) -> EquationNeuron:
    """Return compiler equations that retain every declared initial-state word.

    Parameters
    ----------
    neuron : EquationNeuron
        Authored model. Initial-state entries without equations remain constant
        during integration and can be updated by the model's event reset rules.

    Returns
    -------
    EquationNeuron
        The original model when it has no held state; otherwise a shallow copy
        with independent equation and Jacobian maps. Held words use an identity
        map or a zero derivative. Live state, initial values, compiled Python
        programs and random counters are neither consumed nor modified. This
        copy is a compiler input, not a replacement runtime model.

    Notes
    -----
    Original equation order is retained and held words follow their initial
    declaration order. Exponential Euler receives the held word's exact zero
    diagonal Jacobian. The same preparation feeds native IR, registered and
    folded RTL, and the bit-true C and Rust kernels.
    """
    held = tuple(name for name in neuron.initial_state if name not in neuron.equations)
    if not held:
        return neuron
    lowered = copy(neuron)
    lowered.equations = {
        **neuron.equations,
        **{name: name if neuron.method == "map" else "0" for name in held},
    }
    if neuron.method == "exp_euler":
        lowered.jacobian_expressions = {
            **neuron.jacobian_expressions,
            **{name: "0" for name in held},
        }
    return lowered
