# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo QCFS quantisation and surrogate derivatives

"""Quantise activations and their surrogate derivatives exactly as Python QCFS."""

from std.math import floor, isfinite
from std.memory import Pointer


struct QCFSActivation(Copyable, Movable):
    """Positive step count and finite learned threshold."""

    var steps: Int
    """Number of quantisation intervals and simulation steps."""
    var theta: Float64
    """Upper activation bound and IF threshold."""

    def __init__(out self, steps: Int = 8, theta: Float64 = 1.0) raises:
        """Refuse invalid parameters before constructing an activation.

        Args:
            steps: Positive integer quantisation count.
            theta: Finite positive threshold.

        Raises:
            Error: Nonpositive steps or invalid threshold.
        """
        if steps < 1 or not isfinite(theta) or theta <= 0:
            raise Error("QCFS requires positive steps and a finite positive threshold")
        self.steps = steps
        self.theta = theta

    def forward(self, x: Float64) raises -> Float64:
        """Quantise a scalar, saturating infinities and propagating NaNs.

        Args:
            x: Scalar ANN activation.

        Returns:
            Clipped shifted-grid activation.

        Raises:
            Error: Invalid mutated step count or threshold.
        """
        if self.steps < 1 or not isfinite(self.theta) or self.theta <= 0:
            raise Error("QCFS requires positive steps and a finite positive threshold")
        var steps = Float64(self.steps)
        var shifted = x * steps / self.theta + 0.5
        if shifted < 0:
            shifted = 0
        elif shifted > steps:
            shifted = steps
        return floor(shifted) * self.theta / steps

    def backward(self, x: Float64, upstream: Float64) raises -> Tuple[Float64, Float64]:
        """Return one element's straight-through derivatives (Bu et al., 2022, Eq. 17).

        Args:
            x: Scalar ANN activation.
            upstream: Upstream gradient of the quantised output.

        Returns:
            Input derivative, upstream on the open interior and zero elsewhere,
            and the threshold derivative a one-element batch receives, in the
            Python autograd operation order.

        Raises:
            Error: Invalid mutated step count or threshold.
        """
        if self.steps < 1 or not isfinite(self.theta) or self.theta <= 0:
            raise Error("QCFS requires positive steps and a finite positive threshold")
        var steps = Float64(self.steps)
        var theta = self.theta
        var shifted = x * steps / theta + 0.5
        var interior = shifted > 0 and shifted < steps
        var clipped = shifted
        if clipped < 0:
            clipped = 0
        elif clipped > steps:
            clipped = steps
        var lattice = floor(clipped)
        var output_gradient = upstream / steps
        var carried: Float64 = 0.0
        var retained: Float64 = 0.0
        var input_gradient: Float64 = 0.0
        if interior:
            carried = output_gradient * theta
            retained = x
            input_gradient = carried / theta * steps
        var scaled_input = retained * steps
        var threshold = (0.0 + output_gradient * lattice) + (
            0.0 + (-carried) * (scaled_input / theta / theta)
        )
        return (input_gradient, threshold)


def _span(address: UInt, count: UInt) -> Bool:
    """Admit one complete aligned float64 span in the signed address domain.

    Args:
        address: Array start address.
        count: Number of float64 elements.

    Returns:
        True for an empty span or a nonnull aligned span ending in range.
    """
    if count == 0:
        return True
    var maximum = UInt(0x7FFFFFFFFFFFFFFF)
    return address != 0 and address % 8 == 0 and count <= maximum // 8 and address <= maximum - count * 8


def _read(address: UInt, index: UInt) -> Float64:
    """Read one element of an admitted float64 span.

    Args:
        address: Admitted array start address.
        index: Element index below the admitted count.

    Returns:
        The stored value.
    """
    return Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=Int(address + index * 8))[]


def _write(address: UInt, index: UInt, value: Float64):
    """Write one element of an admitted float64 span.

    Args:
        address: Admitted array start address.
        index: Element index below the admitted count.
        value: Value to store.
    """
    var target = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(address + index * 8))
    target[] = value


@export
def sc_qcfs_abi_version() abi("C") -> UInt32:
    """Report the exact QCFS array ABI version.

    Returns:
        Version one.
    """
    return 1


@export
def sc_qcfs_forward(steps: UInt32, theta: Float64, x_addr: UInt, count: UInt, output_addr: UInt) abi("C") -> Int32:
    """Quantise count activations; every output is unchanged on refusal.

    Args:
        steps: Positive quantisation count.
        theta: Finite positive threshold.
        x_addr: Live aligned float64 input array.
        count: Number of elements.
        output_addr: Live aligned float64 output array; may equal the input.

    Returns:
        Zero on success or minus one for invalid parameters or spans.
    """
    if steps == 0 or not isfinite(theta) or theta <= 0:
        return -1
    if not _span(x_addr, count) or not _span(output_addr, count):
        return -1
    try:
        var activation = QCFSActivation(Int(steps), theta)
        for index in range(count):
            _write(output_addr, index, activation.forward(_read(x_addr, index)))
        return 0
    except:
        return -1


@export
def sc_qcfs_backward(
    steps: UInt32,
    theta: Float64,
    x_addr: UInt,
    upstream_addr: UInt,
    count: UInt,
    input_gradient_addr: UInt,
    threshold_gradient_addr: UInt,
) abi("C") -> Int32:
    """Write each element's input and threshold derivative; outputs unchanged on refusal.

    Args:
        steps: Positive quantisation count.
        theta: Finite positive threshold.
        x_addr: Live aligned float64 input array.
        upstream_addr: Live aligned float64 upstream gradients.
        count: Number of elements.
        input_gradient_addr: Live aligned float64 input-derivative output.
        threshold_gradient_addr: Live aligned float64 threshold-derivative output
            that must not overlap the input-derivative output.

    Returns:
        Zero on success or minus one for invalid parameters or spans.
    """
    if steps == 0 or not isfinite(theta) or theta <= 0:
        return -1
    if not (
        _span(x_addr, count)
        and _span(upstream_addr, count)
        and _span(input_gradient_addr, count)
        and _span(threshold_gradient_addr, count)
    ):
        return -1
    try:
        var activation = QCFSActivation(Int(steps), theta)
        for index in range(count):
            var derivatives = activation.backward(_read(x_addr, index), _read(upstream_addr, index))
            _write(input_gradient_addr, index, derivatives[0])
            _write(threshold_gradient_addr, index, derivatives[1])
        return 0
    except:
        return -1


