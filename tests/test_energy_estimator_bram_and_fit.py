# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA estimator block-RAM units and target fit

"""Block-RAM units, the full target fit and input checks of the two FPGA estimators.

The FPGA targets list block RAM in kilobits; the estimate reports give ``total_bram_kb`` in
kilobytes. The fit must compare like with like (bits), include the block RAM and the clock as
well as the LUTs, and an unknown target or a non-positive clock must be refused rather than
reported as fitting. The edge power/thermal payload, whose target block RAM changed unit, is
schema v2.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest

from sc_neurocore.edge.power_thermal import (
    SCHEMA_VERSION,
    PowerThermalConfig,
    build_power_thermal_model,
)
from sc_neurocore.energy.estimator import EnergyReport, LayerEstimate, estimate
from sc_neurocore.energy.folded_estimator import FoldedAreaEstimate
from sc_neurocore.energy.fpga_models import ICE40_HX8K, TARGETS, FPGATarget


def _layer(*, luts: int = 100, bram_bits: int = 0, power_mw: float = 1.0) -> LayerEstimate:
    """Return one small layer estimate with the given LUTs, block RAM and power."""
    return LayerEstimate(
        name="layer_0",
        n_inputs=4,
        n_neurons=4,
        n_synapses=16,
        bitstream_length=8,
        luts=luts,
        ffs=10,
        bram_bits=bram_bits,
        dynamic_power_mw=power_mw,
        latency_cycles=10,
    )


def _folded(
    *, bram_bits: int, clock_mhz: float = 100.0, target: str = "ice40"
) -> FoldedAreaEstimate:
    """Return a small folded estimate with the given block RAM, clock and target."""
    return FoldedAreaEstimate(
        target=target,
        pe_luts=100,
        multiplier_luts=0,
        rom_luts=0,
        infra_luts=0,
        dsps=0,
        bram_bits=bram_bits,
        latency_cycles=10,
        clock_freq_mhz=clock_mhz,
        dynamic_power_mw=1.0,
    )


def test_target_block_ram_is_held_in_kilobits_and_exposed_in_bits_and_kilobytes() -> None:
    """The iCE40 HX8K lists 128 Kbit: 131072 bits, 16 KB."""
    assert ICE40_HX8K.total_bram_kbits == 128
    assert ICE40_HX8K.total_bram_bits == 131072
    assert ICE40_HX8K.total_bram_kb == 16.0
    for target in TARGETS.values():
        assert target.total_bram_kb == target.total_bram_kbits / 8


def test_block_ram_over_the_target_does_not_fit() -> None:
    """One bit more block RAM than the target holds fails the fit, even with few LUTs."""
    within = EnergyReport(target="ice40", layers=[_layer(bram_bits=ICE40_HX8K.total_bram_bits)])
    over = EnergyReport(target="ice40", layers=[_layer(bram_bits=ICE40_HX8K.total_bram_bits + 1)])
    assert within.fits_on_target
    assert not over.fits_on_target
    assert "BRAM" in over.summary()


def test_folded_estimate_compares_block_ram_in_bits() -> None:
    """200000 bits (24.4 KB) exceed the HX8K's 131072 bits although 24.4 is below 128."""
    over = _folded(bram_bits=200_000)
    assert over.total_bram_kb == pytest.approx(200_000 / 8192.0)
    assert not over.fits_on_target
    assert _folded(bram_bits=131_072).fits_on_target


def test_a_clock_above_the_target_maximum_does_not_fit() -> None:
    """A 200 MHz clock exceeds the HX8K's 150 MHz in both estimators."""
    report = EnergyReport(target="ice40", layers=[_layer()], clock_freq_mhz=200.0)
    assert not report.fits_on_target
    assert "clock" in report.summary()
    assert EnergyReport(target="ice40", layers=[_layer()], clock_freq_mhz=150.0).fits_on_target
    assert not _folded(bram_bits=0, clock_mhz=200.0).fits_on_target


@pytest.mark.parametrize("clock", [0.0, -5.0, float("nan"), float("inf")])
def test_reports_refuse_a_non_positive_or_non_finite_clock(clock: float) -> None:
    """A clock that is not finite and positive is refused, not divided by."""
    with pytest.raises(ValueError, match="clock_freq_mhz"):
        EnergyReport(target="ice40", layers=[_layer()], clock_freq_mhz=clock)
    with pytest.raises(ValueError, match="clock_freq_mhz"):
        _folded(bram_bits=0, clock_mhz=clock)
    with pytest.raises(ValueError, match="clock_mhz"):
        estimate([(4, 4)], clock_mhz=clock)


def test_reports_refuse_an_unknown_target() -> None:
    """An unknown target is refused instead of being reported as fitting at 0 %."""
    with pytest.raises(ValueError, match="Unknown target"):
        EnergyReport(target="unknown", layers=[_layer()])
    with pytest.raises(ValueError, match="Unknown target"):
        _folded(bram_bits=0, target="unknown")


def test_neuron_type_event_is_the_event_driven_architecture() -> None:
    """neuron_type='event' gives the same estimate as event_driven=True."""
    by_type = estimate([(16, 8)], neuron_type="event")
    by_flag = estimate([(16, 8)], event_driven=True)
    assert by_type.total_luts == by_flag.total_luts
    assert by_type.total_dynamic_power_mw == by_flag.total_dynamic_power_mw
    assert by_type.total_luts != estimate([(16, 8)]).total_luts


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"neuron_type": "izhikevich"}, "neuron_type"),
        ({"bitstream_length": 0}, "bitstream_length"),
        ({"layer_sizes": [(0, 4)]}, "layer sizes"),
        ({"layer_sizes": [(4, -1)]}, "layer sizes"),
    ],
)
def test_estimate_refuses_invalid_arguments(kwargs: dict[str, object], match: str) -> None:
    """An unknown neuron type, a zero bitstream or a non-positive layer size is refused."""
    arguments: dict[str, object] = {"layer_sizes": [(4, 4)], **kwargs}
    with pytest.raises(ValueError, match=match):
        cast(Any, estimate)(**arguments)


def test_infrastructure_luts_draw_dynamic_power() -> None:
    """With the AXI infrastructure included, its LUTs add to the dynamic power."""
    with_infra = estimate([(16, 8)], include_infra=True)
    without = estimate([(16, 8)], include_infra=False)
    layers_power = sum(layer.dynamic_power_mw for layer in with_infra.layers)
    assert with_infra.infra_luts > 0
    assert with_infra.infra_dynamic_power_mw > 0.0
    assert with_infra.total_dynamic_power_mw == pytest.approx(
        layers_power + with_infra.infra_dynamic_power_mw
    )
    assert without.total_dynamic_power_mw == pytest.approx(layers_power)


def test_luts_over_the_target_do_not_fit_and_are_named() -> None:
    """One LUT more than the target holds fails the fit, and the summary names the LUT budget."""
    over = EnergyReport(target="ice40", layers=[_layer(luts=ICE40_HX8K.total_luts + 1)])
    assert not over.fits_on_target
    assert "exceeds the LUT budget" in over.summary()


def test_the_old_kilobyte_field_name_is_refused() -> None:
    """``FPGATarget(total_bram_kb=...)`` fails: the field is ``total_bram_kbits``."""
    fields = {f.name: getattr(ICE40_HX8K, f.name) for f in dataclasses.fields(FPGATarget)}
    fields["total_bram_kb"] = fields.pop("total_bram_kbits")
    with pytest.raises(TypeError, match="total_bram_kb"):
        cast(Any, FPGATarget)(**fields)


def test_power_thermal_payload_is_v2_with_block_ram_in_kilobytes_and_kilobits() -> None:
    """The v2 payload gives the target block RAM in kilobytes and kilobits."""
    payload = build_power_thermal_model(PowerThermalConfig(target="ice40", layer_sizes=((4, 4),)))
    assert SCHEMA_VERSION == "sc-neurocore.fpga.power-thermal.v2"
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["target"]["total_bram_kbits"] == 128
    assert payload["target"]["total_bram_kb"] == 16.0
