# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo checked dense IF numeric reservation

"""Checked nonnegative host integer arithmetic for dense IF numeric buffers."""


def checked_add(a: Int, b: Int) raises -> Int:
    """Add admitted nonnegative sizes, refusing address overflow.

    Args:
        a: First nonnegative numeric element count.
        b: Second nonnegative numeric element count.

    Returns:
        Addressable sum without native integer wrapping.

    Raises:
        Error: Negative count or native address overflow.
    """
    if a < 0 or b < 0 or a > 0x7FFFFFFFFFFFFFFF - b:
        raise Error("IF resource limit")
    return a + b


def checked_mul(a: Int, b: Int) raises -> Int:
    """Multiply admitted nonnegative sizes, refusing address overflow.

    Args:
        a: First nonnegative numeric extent.
        b: Second nonnegative numeric extent.

    Returns:
        Addressable product without native integer wrapping.

    Raises:
        Error: Negative extent or native address overflow.
    """
    if a < 0 or b < 0 or (a != 0 and b > 0x7FFFFFFFFFFFFFFF // a):
        raise Error("IF resource limit")
    return a * b


def admit_copy(coefficients: Int, maximum: Int) raises:
    """Reserve sixteen bytes per coefficient before taking owned copies.

    Args:
        coefficients: Nonnegative total weight and bias count.
        maximum: Positive numeric byte limit excluding caller/runtime storage.

    Raises:
        Error: Invalid budget, native address overflow or exceeded reservation.
    """
    if maximum <= 0:
        raise Error("IF invalid input")
    if checked_mul(coefficients, 16) > maximum:
        raise Error("IF resource limit")


def admit_replay(coefficients: Int, inputs: Int, widths: List[Int], steps: Int, batch: Int, trace: Bool, linear: Bool, maximum: Int) raises:
    """Admit 8*(2P+2F+2S+2O+H+5M+I), excluding caller/runtime storage.

    Args:
        coefficients: Nonnegative weight and bias count P.
        inputs: Positive first-layer input width.
        widths: Nonempty checked positive weighted output widths.
        steps: Nonnegative explicit input timestep count.
        batch: Nonnegative sample count.
        trace: Include complete state/event storage H.
        linear: Exclude final linear stage from event storage.
        maximum: Positive numeric byte limit.

    Raises:
        Error: Invalid budget, address overflow or exceeded numeric reservation.
    """
    if maximum <= 0:
        raise Error("IF invalid input")
    var nodes = 0
    var largest = 0
    for width in widths:
        nodes = checked_add(nodes, width)
        if width > largest:
            largest = width
    var states = checked_mul(batch, nodes)
    var output = checked_mul(batch, widths[len(widths)-1])
    var frame = checked_mul(batch, inputs)
    var frames = checked_mul(steps, frame)
    var total = checked_mul(checked_add(checked_add(coefficients, frames), checked_add(states, output)), 2)
    if trace:
        var spiking = nodes
        if linear:
            spiking -= widths[len(widths)-1]
        total = checked_add(total, checked_mul(checked_mul(steps, batch), checked_add(nodes, spiking)))
    total = checked_add(total, checked_mul(checked_mul(batch, largest), 5))
    total = checked_add(total, frame)
    if checked_mul(total, 8) > maximum:
        raise Error("IF resource limit")
