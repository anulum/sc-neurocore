# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The deployment blob's weight codes at every device precision

"""Hold the deployment package's weight codes, their scale and its placement checks.

A 1-bit device once quantised every synapse weight to zero, a precision above 16 bits overflowed the blob's 16-bit
weight field with ``struct.error``, the scale that turns codes back into weights was never recorded, an empty
placement list crashed packaging, and validation accepted negative core and local ids. These tests hold the binary
codes of one bit, the refusal of precisions the field cannot hold and of non-finite weights, the recorded scale,
an empty placement list, and the refusal of negative ids.
"""

from __future__ import annotations

import struct
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from sc_neurocore.hardware import Deployer, DeviceFamily, NeuronPlacement, get_device


def _synapse_codes(blob: bytes, n_placements: int) -> list[tuple[int, int, int]]:
    """Return the (source, target, code) rows of a blob's synapse table."""
    _, _, n_synapses = struct.unpack_from(">4sII", blob, 0)
    offset = 12 + 8 * n_placements
    return [struct.unpack_from(">IIh", blob, offset + 10 * i) for i in range(n_synapses)]


def _placements(n: int) -> list[NeuronPlacement]:
    """Place ``n`` neurons on one core, one per local slot."""
    return [NeuronPlacement(neuron_id=i, core_id=0, local_id=i) for i in range(n)]


def _weights() -> np.ndarray[Any, Any]:
    """Return a 3 x 3 weight matrix with signed non-zero entries and zeros."""
    return np.array([[0.0, 0.5, -0.25], [1.0, 0.0, -2.0], [0.0, 0.75, 0.0]])


def test_one_bit_codes_carry_the_sign_and_the_scale_recovers_the_mean_magnitude() -> None:
    """One bit stores +1 or -1 per synapse; code / scale is plus or minus the mean non-zero magnitude."""
    w = _weights()
    device = replace(get_device(DeviceFamily.LOIHI), weight_bits=1)
    pkg = Deployer().package(w, device, _placements(3))
    rows = _synapse_codes(pkg.config_blob, 3)
    codes = {(src, tgt): code for src, tgt, code in rows}
    assert set(codes.values()) == {-1, 1}
    for (src, tgt), code in codes.items():
        assert code == (1 if w[src, tgt] > 0 else -1)
    level = float(np.mean(np.abs(w[w != 0])))
    assert pkg.metadata["weight_scale"] == pytest.approx(1.0 / level)


@pytest.mark.parametrize("bits", [2, 8, 16])
def test_multi_bit_codes_decode_within_half_a_step(bits: int) -> None:
    """With ``b`` bits the codes span ``2**(b-1) - 1`` levels a side, and code / scale is within half a step."""
    w = _weights()
    device = replace(get_device(DeviceFamily.LOIHI), weight_bits=bits)
    pkg = Deployer().package(w, device, _placements(3))
    scale = pkg.metadata["weight_scale"]
    w_max = 2 ** (bits - 1) - 1
    assert scale == pytest.approx(w_max / 2.0)
    for src, tgt, code in _synapse_codes(pkg.config_blob, 3):
        assert abs(code) <= w_max
        assert abs(code / scale - w[src, tgt]) <= 0.5 / scale + 1e-12


@pytest.mark.parametrize("bits", [0, 17, 32])
def test_a_precision_the_16_bit_weight_field_cannot_hold_is_refused(bits: int) -> None:
    """Below one bit no weight can be coded; above 16 bits the blob's signed 16-bit field would overflow."""
    device = replace(get_device(DeviceFamily.LOIHI), weight_bits=bits)
    with pytest.raises(ValueError, match="weight_bits"):
        Deployer().package(_weights(), device, _placements(3))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_weights_are_refused(bad: float) -> None:
    """A weight that is NaN or infinite has no code."""
    w = _weights()
    w[0, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        Deployer().package(w, get_device(DeviceFamily.LOIHI), _placements(3))


def test_an_empty_placement_list_packages_with_no_core_used() -> None:
    """Packaging no placement uses no core instead of failing on an empty maximum."""
    pkg = Deployer().package(np.zeros((0, 0)), get_device(DeviceFamily.LOIHI), [])
    assert pkg.metadata["n_cores_used"] == 0
    assert pkg.metadata["fits"] is True
    assert Deployer().validate(pkg) is True


@pytest.mark.parametrize("field", ["neuron_id", "core_id", "local_id"])
def test_validation_refuses_a_negative_id(field: str) -> None:
    """A negative neuron, core or local id names no slot on the device."""
    deployer = Deployer()
    pkg = deployer.package(_weights(), get_device(DeviceFamily.LOIHI), _placements(3))
    assert deployer.validate(pkg) is True
    setattr(pkg.placements[1], field, -1)
    assert deployer.validate(pkg) is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("neuron_id", -1),
        ("neuron_id", 2**32),
        ("core_id", -1),
        ("core_id", 2**16),
        ("local_id", -1),
        ("local_id", 2**16),
    ],
)
def test_packaging_refuses_a_placement_id_its_field_cannot_hold(field: str, value: int) -> None:
    """A negative id, or one wider than its unsigned field, is refused before the blob is packed."""
    placements = _placements(3)
    setattr(placements[1], field, value)
    with pytest.raises(ValueError, match=rf"placement {field}={value} "):
        Deployer().package(_weights(), get_device(DeviceFamily.LOIHI), placements)


def test_packaging_takes_the_widest_ids_each_field_can_hold() -> None:
    """The largest neuron, core and local ids still pack."""
    placements = _placements(3)
    placements[2].neuron_id = 2**32 - 1
    placements[2].core_id = 2**16 - 1
    placements[2].local_id = 2**16 - 1
    pkg = Deployer().package(_weights(), get_device(DeviceFamily.LOIHI), placements)
    assert struct.unpack_from(">IHH", pkg.config_blob, 12 + 2 * 8) == (
        2**32 - 1,
        2**16 - 1,
        2**16 - 1,
    )
